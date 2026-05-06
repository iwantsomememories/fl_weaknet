import argparse
import csv
from statistics import mode
import sys
import random
import time
import os

from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np

sys.path.append("../../")

from utils.ComunicationDelay import MultiModeDelayGenerator, DelayGenerator
from utils.Datasets import CompletePartitionedMNIST

from fedlab.core.client.manager import ClientManager, PassiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.utils import MessageCode, Logger, SerializationTool
from fedlab.models.mlp import MLP
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.contrib.algorithm.fedavg import FedAvgClientTrainer
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST


class WeaknetTrace:
    """CSV-backed weak-network trace indexed by (round, client_id)."""

    def __init__(self, path: str):
        self.path = path
        self.events = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {
                "round",
                "client_id",
                "total_delay",
                "lost",
                "disconnected",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "Trace file {} missing columns: {}".format(
                        path, sorted(missing)))

            for row in reader:
                trace_round = int(row["round"])
                client_id = int(row["client_id"])
                self.events[(trace_round, client_id)] = {
                    "total_delay": float(row["total_delay"]),
                    "lost": int(row["lost"]),
                    "disconnected": int(row["disconnected"]),
                    "event_type": row.get("event_type", "unknown"),
                }

    def get(self, trace_round: int, client_id: int):
        return self.events.get((trace_round, client_id))


class SemiAsyncClientTrainer(SGDClientTrainer):

    def __init__(self,
                model:torch.nn.Module,
                cuda:bool=False,
                device:str=None,
                logger:Logger=None,
                use_usce: bool=False,
                usce_beta0: float=0.1,
                usce_kappa: float=1.0,
                usce_rce_label_min: float=1e-4,
                usce_eps: float=1e-12):
        super().__init__(model, cuda, device, logger)
        self.model_version = None
        self.system_uncertainty = 0.0
        self.avg_train_loss = 0.0
        self.use_usce = use_usce
        self.usce_beta0 = usce_beta0
        self.usce_kappa = usce_kappa
        self.usce_rce_label_min = usce_rce_label_min
        self.usce_eps = usce_eps
    
    @property
    def uplink_package(self):
        return [
            self.model_parameters,
            self.model_version,
            torch.tensor(self.data_size).to(self.model_parameters.dtype),
            torch.tensor(self.avg_train_loss).to(self.model_parameters.dtype)
        ]

    def setup_dataset(self, dataset: CompletePartitionedMNIST, client_id):
        self.dataset = dataset
        self.data_size = len(dataset.get_dataset(cid=client_id, type="train"))
        self.data_id = client_id
        self._LOGGER.info("Data size of {}: {}".format(client_id, self.data_size))
    
    def local_process(self, payload):
        model_parameters = payload[0]
        self.model_version = payload[1]
        self.system_uncertainty = payload[2] if len(payload) > 2 else 0.0

        # # print("model_parameters: ", model_parameters)
        # print("Client {} received model version {}".format(id, self.model_version))

        train_loader = self.dataset.get_dataloader(cid=self.data_id, batch_size=self.batch_size)

        # print("Client {} is training...".format(id))
        self.train(model_parameters, train_loader)
        # print("Client {} finished training.".format(id))
    
    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        total_loss = 0.0
        total_samples = 0
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                if self.use_usce:
                    loss = self.uncertainty_aware_sce_loss(
                        outputs, target, self.system_uncertainty)
                else:
                    loss = self.criterion(outputs, target)

                batch_size = target.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if total_samples > 0:
            self.avg_train_loss = total_loss / total_samples
        self._LOGGER.info("Local train procedure is finished")

    def uncertainty_aware_sce_loss(self, outputs, target, system_uncertainty):
        """Uncertainty-aware symmetric cross entropy loss."""
        num_classes = outputs.size(1)
        probs = F.softmax(outputs, dim=1)

        ce = F.cross_entropy(outputs, target, reduction="none")

        one_hot = F.one_hot(target, num_classes=num_classes).to(
            device=outputs.device, dtype=outputs.dtype)
        clipped_labels = torch.clamp(one_hot, min=self.usce_rce_label_min, max=1.0)
        rce = -torch.sum(probs * torch.log(clipped_labels), dim=1)

        entropy = -torch.sum(
            probs.detach() * torch.log(probs.detach().clamp_min(self.usce_eps)),
            dim=1
        )
        normalized_entropy = entropy / torch.log(
            torch.tensor(num_classes, device=outputs.device, dtype=outputs.dtype))

        system_uncertainty = torch.as_tensor(
            system_uncertainty, device=outputs.device, dtype=outputs.dtype)
        beta = self.usce_beta0 * torch.exp(
            -self.usce_kappa * system_uncertainty) * normalized_entropy
        beta = torch.clamp(beta, min=0.0, max=1.0)
        alpha = 1.0 - beta

        return torch.mean(alpha * ce + beta * rce)

class SemiAsyncClientManager(ClientManager):

    def __init__(self,
                network: DistNetwork,
                trainer: ModelMaintainer,
                logger: Logger=None,
                delay_gen: DelayGenerator=None,
                trace: WeaknetTrace=None,
                trace_round_offset: int=1):
        super().__init__(network, trainer)
        self.delay_gen = delay_gen
        self.trace = trace
        self.trace_round_offset = trace_round_offset
        self._LOGGER = Logger() if logger is None else logger
    
    def main_loop(self):
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:
                # id_list, payload = payload[0].to(
                #     torch.int32).tolist(), payload[1:]

                # assert len(id_list) == 1
                self._trainer.local_process(payload=payload)
                self._LOGGER.info(
                    "Finished local process, model version: {}, system uncertainty: {}".format(
                        self._trainer.model_version, self._trainer.system_uncertainty))

                self.synchronize()

            else:
                raise ValueError(
                    "Invalid MessageCode {}. Please check MessageCode list.".
                    format(message_code))

    def synchronize(self):
        self._LOGGER.info("Uploading information to server.")

        trace_event = None
        client_id = self._network.rank - 1
        if self.trace is not None:
            trace_round = int(self._trainer.model_version) + self.trace_round_offset
            trace_event = self.trace.get(trace_round, client_id)

        if trace_event is not None:
            if trace_event["lost"] == 1 or trace_event["disconnected"] == 1:
                self._LOGGER.info(
                    "Trace drops client {} update at trace round {} "
                    "(event_type: {}).".format(
                        client_id, trace_round, trace_event["event_type"]))
                return
            actual_delay = trace_event["total_delay"]
            self._LOGGER.info(
                "Trace delay for client {} at trace round {}: {:.4f}s "
                "(event_type: {}).".format(
                    client_id, trace_round, actual_delay,
                    trace_event["event_type"]))
        elif self.delay_gen is not None:
            # 模拟通信时延
            actual_delay = self.delay_gen.generate_delay()
        else:
            actual_delay = 0.0

        time.sleep(actual_delay)

        self._network.send(content=self._trainer.uplink_package,
                            message_code=MessageCode.ParameterUpdate,
                            dst=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-async Client")

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3009')
    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--rank', type=int)

    # 本地训练参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_usce", action="store_true")
    parser.add_argument("--usce_beta0", type=float, default=0.1)
    parser.add_argument("--usce_kappa", type=float, default=1.0)
    parser.add_argument("--usce_rce_label_min", type=float, default=1e-4)

    # 模型与数据集划分
    parser.add_argument('--model', type=str, default="mnist")
    parser.add_argument('--partition', type=str, default="iid", choices=["iid", "noniid-labeldir"])
    parser.add_argument('--dir_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--trace_path", type=str, default=None)
    parser.add_argument(
        "--trace_round_offset",
        type=int,
        default=1,
        help="Trace round = received model_version + trace_round_offset.")

    args = parser.parse_args()

    if args.model == "mnist":
        model = CNN_MNIST()
        if args.partition == "iid":
            dataset = CompletePartitionedMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients, partition="iid")
        else:
            dataset = CompletePartitionedMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients,  partition="noniid-labeldir", dir_alpha=args.dir_alpha, seed=args.seed)
    else:
        raise NotImplementedError


    network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank, ethernet=args.ethernet)

    # logs_path = "../logs/"
    # if not os.path.exists(logs_path):
    #     os.makedirs(logs_path)
    # LOGGER = Logger(log_name="semiasync_client " + str(args.rank), log_file="../logs/semiasync_client_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".log")

    trainer = SemiAsyncClientTrainer(
        model,
        cuda=torch.cuda.is_available(),
        use_usce=args.use_usce,
        usce_beta0=args.usce_beta0,
        usce_kappa=args.usce_kappa,
        usce_rce_label_min=args.usce_rce_label_min)

    print("rank: ", args.rank)
    
    trainer.setup_dataset(dataset, args.rank - 1)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    factory = MultiModeDelayGenerator()
    
    trace = WeaknetTrace(args.trace_path) if args.trace_path else None

    manager = SemiAsyncClientManager(
        network=network,
        trainer=trainer,
        delay_gen=factory.get_generator(),
        trace=trace,
        trace_round_offset=args.trace_round_offset)
    manager.run()

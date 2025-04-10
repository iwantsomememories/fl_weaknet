import argparse
from statistics import mode
import sys
import random
import time
import os

from datetime import datetime

import torch
from torch import nn
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



class SemiAsyncClientTrainer(SGDClientTrainer):

    def __init__(self,
                model:torch.nn.Module,
                cuda:bool=False,
                device:str=None,
                logger:Logger=None):
        super().__init__(model, cuda, device, logger)
        self.model_version = None
    
    @property
    def uplink_package(self):
        return [self.model_parameters, self.model_version, torch.tensor(self.data_size).to(self.model_parameters.dtype)]

    def setup_dataset(self, dataset: CompletePartitionedMNIST, client_id):
        self.dataset = dataset
        self.data_size = len(dataset.get_dataset(cid=client_id, type="train"))
        self.data_id = client_id
        self._LOGGER.info("Data size of {}: {}".format(client_id, self.data_size))
    
    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.model_version = payload[1]

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
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")

class SemiAsyncClientManager(ClientManager):

    def __init__(self,
                network: DistNetwork,
                trainer: ModelMaintainer,
                logger: Logger=None,
                delay_gen: DelayGenerator=None):
        super().__init__(network, trainer)
        self.delay_gen = delay_gen
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
                self._trainer.local_process(payload=payload, id=self._trainer.data_id)
                self._LOGGER.info("Finished local process, model version: {}".format(self._trainer.model_version))

                self.synchronize()

            else:
                raise ValueError(
                    "Invalid MessageCode {}. Please check MessageCode list.".
                    format(message_code))

    def synchronize(self):
        self._LOGGER.info("Uploading information to server.")

        # 模拟通信时延
        actual_delay = self.delay_gen.generate_delay()
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

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument('--model', type=str, default="mnist")
    parser.add_argument('--partition', type=str, default="iid", choices=["iid", "noniid-labeldir"])
    parser.add_argument('--dir_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)

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

    trainer = SemiAsyncClientTrainer(model, cuda=torch.cuda.is_available())

    print("rank: ", args.rank)
    trainer.setup_dataset(dataset, args.rank - 1)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    factory = MultiModeDelayGenerator()
    
    manager = SemiAsyncClientManager(network=network, trainer=trainer, delay_gen=factory.get_generator())
    manager.run()
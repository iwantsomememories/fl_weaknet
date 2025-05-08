import torch
from torch import nn
import time

from fedlab.core.client.manager import ActiveClientManager, ClientManager
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.network import DistNetwork
from fedlab.utils import MessageCode, Logger, SerializationTool
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST
import argparse

from utils.Datasets import CompletePartitionedMNIST
from utils.ComunicationDelay import MultiModeDelayGenerator, DelayGenerator


class ASTrainer(SGDClientTrainer):
    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]

    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.round = payload[1]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

class AsyncClientManager(ClientManager):
    def __init__(self,
                 network: DistNetwork,
                 trainer: ClientTrainer,
                 logger: Logger = None,
                 delay_gen: DelayGenerator=None):
        super().__init__(network, trainer)
        self.delay_gen = delay_gen
        self._LOGGER = Logger() if logger is None else logger

    def main_loop(self):
        """Actions to perform on receiving new message, including local training.

            1. client requests data from server (ACTIVELY).
            2. after receiving data, client will train local model.
            3. client will synchronize with server actively.
        """
        while True:
            # request model actively
            self.request()

            # waits for data from server
            _, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:
                self._trainer.local_process(id=self._network.rank-1, payload=payload)
                self.synchronize()

            else:
                raise ValueError(
                    "Invalid MessageCode {}. Please check MessageCode Enum.".
                    format(message_code))

    def request(self):
        """Client request."""
        self._LOGGER.info("request parameter procedure.")
        self._network.send(message_code=MessageCode.ParameterRequest, dst=0)

    def synchronize(self):
        # 模拟通信时延
        actual_delay = self.delay_gen.generate_delay()
        time.sleep(actual_delay)

        self._LOGGER.info("Uploading information to server.")
        self._network.send(content=self._trainer.uplink_package,
                           message_code=MessageCode.ParameterUpdate,
                           dst=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async FL cross process")

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument("--ethernet", type=str, default=None)

    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--rank', type=int)

    # 本地训练参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)

    # 模型与数据集划分
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
    
    trainer = ASTrainer(model, cuda=torch.cuda.is_available())
    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank, ethernet=args.ethernet)
    
    manager = ActiveClientManager(trainer=trainer, network=network)
    manager.run()
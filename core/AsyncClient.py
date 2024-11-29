import torch
from torch import nn
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_client import SGDClientTrainer
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST
import argparse


class ASTrainer(SGDClientTrainer):
    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]

    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.round = payload[1]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async FL cross process")
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--rank', type=int)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--model', type=str, default="mnist")

    args = parser.parse_args()

    if args.model == "mnist":
        model = CNN_MNIST()
        dataset = PathologicalMNIST(root='../datasets/mnist/', path="../datasets/mnist/")
    else:
        raise NotImplementedError
    
    trainer = ASTrainer(model, cuda=False)
    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank, ethernet="eth1")
    
    manager = ActiveClientManager(trainer=trainer, network=network)
    manager.run()
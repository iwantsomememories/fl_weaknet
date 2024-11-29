import os
import torch
from torch import nn
from typing import List
from fedlab.utils.logger import Logger
from fedlab.utils.functional import evaluate
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_server import AsyncServerHandler, ServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST

import h5py
import argparse

class ASHandler(AsyncServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int,
        cuda: bool = False,
        device: str = None,
        logger: Logger = None,
        evaluate_gap: int = 5,
    ):
        super(ASHandler, self).__init__(model, global_round, num_clients, cuda, device, logger)
        self.evaluate_gap = evaluate_gap
        self.test_loss = []
        self.test_acc = []

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset
    
    def evaluate(self):
        self._model.eval()
        test_loader = self.dataset.get_dataloader(type="test", batch_size=128)
        loss_, acc_ = evaluate(self._model, nn.CrossEntropyLoss(), test_loader)
        self._LOGGER.info(
            f"Round [{self.round}/{self.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}%"
        )

        return loss_, acc_
    
    def load(self, payload: List[torch.Tensor]) -> bool:
        self.global_update(payload)
        self.round += 1
        if self.round % self.evaluate_gap == 0:
            loss, acc = self.evaluate()
            self.test_loss.append((self.round, loss))
            self.test_acc.append((self.round, acc))
    
class ASManager(AsynchronousServerManager):
    def __init__(self,
                 network: DistNetwork,
                 handler: ServerHandler,
                 logger: Logger=None,
                 dataset_name: str='mnist'):
        super(ASManager, self).__init__(network, handler, logger)
        self.dataset_name = dataset_name
    
    def save_results(self):
        algo = "_".join(["async", self.dataset_name, str(self._handler.num_clients), str(self._handler.global_round)])

        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        if len(self._handler.test_acc) > 0:
            file_path = result_path + "{}.h5".format(algo)
            self._LOGGER.info("File path: " + file_path)
        
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('test_acc', data=self._handler.test_acc)
            hf.create_dataset('test_loss', data=self._handler.test_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async FL cross process")
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--model', type=str, default="mnist")
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    if args.model == "mnist":
        model = CNN_MNIST()
        dataset = PathologicalMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients)
        dataset.preprocess()
    else:
        raise NotImplementedError
    
    handler = ASHandler(model, args.global_round, args.num_clients)
    handler.setup_optim(args.alpha)
    handler.setup_dataset(dataset)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0, ethernet="eth1")
    
    manager = ASManager(handler=handler, network=network)

    manager.run()
    manager.save_results()




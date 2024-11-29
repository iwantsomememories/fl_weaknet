from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch

sys.path.append("../../")
torch.manual_seed(0)

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate, get_best_gpu

from fedlab.models import CNN_MNIST
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.core.server.handler import ServerHandler

import h5py

class StandalonePipeline(object):
    def __init__(self, handler: ServerHandler, trainer: SerialClientTrainer, evaluate_gap = 5, datasetname = "mnist"):
        """Perform standalone simulation process.

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer

        # initialization
        self.handler.num_clients = self.trainer.num_clients
        self.evaluate_gap = evaluate_gap
        self.test_loss = []
        self.test_acc = []
        self.datasetname = datasetname

    def save_results(self):
        algo = "_".join(["sync", self.dataset_name, str(self.handler.num_clients), str(self.handler.global_round)])

        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        if len(self.test_acc) > 0:
            file_path = result_path + "{}.h5".format(algo)
            self._LOGGER.info("File path: " + file_path)
        
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('test_acc', data=self.test_acc)
            hf.create_dataset('test_loss', data=self.test_loss)

    def main(self):
        while self.handler.if_stop is False:
            # server side
            # sampled_clients = self.handler.sample_clients()
            selected_clients = [x for x in range(self.handler.num_clients)]
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, selected_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            if self.handler.round % self.evaluate_gap == 0:
                loss, acc = self.handler.evaluate()
                self.test_loss.append((self.handler.round, loss))
                self.test_acc.append((self.handler.round, acc))
            # self.handler.evaluate()
        
        self.save_results()
        

if __name__ == "__main__":
    # configuration
    parser = argparse.ArgumentParser(description="Standalone training example")
    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--model', type=str, default="mnist")

    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    if args.model == "mnist":
        model = CNN_MNIST()
        dataset = PathologicalMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients)
        dataset.preprocess()
    else:
        raise NotImplementedError

    # server
    handler = SyncServerHandler(
        model, args.global_round, args.sample_ratio
    )

    # client
    trainer = SGDSerialClientTrainer(model, args.num_clients, cuda=False)
    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    handler.num_clients = args.num_clients
    handler.setup_dataset(dataset)
    # main
    pipeline = StandalonePipeline(handler, trainer)
    pipeline.main()

import os
import torch
from torch import nn
from torch.multiprocessing import Queue
from typing import List
import threading
import time
from datetime import datetime

from fedlab.utils import MessageCode, Logger
from fedlab.utils.functional import evaluate
from fedlab.core.network import DistNetwork
from fedlab.contrib.algorithm.basic_server import AsyncServerHandler, ServerHandler
from fedlab.core.server.manager import AsynchronousServerManager, ServerManager
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST

from utils.Datasets import CompletePartitionedMNIST

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
        target_accuracy: float = None,
        last_test_acc: float = 0.0,
    ):
        super(ASHandler, self).__init__(model, global_round, num_clients, cuda, device, logger)
        self.evaluate_gap = evaluate_gap
        self.test_loss = []
        self.test_acc = []

        # 终止条件
        self.target_accuracy = target_accuracy
        self.last_test_acc = last_test_acc

    @property
    def if_stop(self):
        if self.target_accuracy is not None:
            # 终止条件：达到目标精度
            return self.last_test_acc >= self.target_accuracy or self.round >= self.global_round
        # 终止条件：达到最大轮数
        return self.round >= self.global_round

    def setup_dataset(self, dataset) -> None:
        self.dataset = dataset
    
    def evaluate(self, start_time=None) -> float:
        self._model.eval()
        test_loader = self.dataset.get_dataloader(type="test", batch_size=128)
        loss_, acc_ = evaluate(self._model, nn.CrossEntropyLoss(), test_loader)

        if self.target_accuracy is not None:
            self.last_test_acc = acc_ * 100.0

        cur_time = time.time() - start_time
        self._LOGGER.info(
            f"Round [{self.round}/{self.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}% Curtime: {cur_time:.2f}s"
        )

        return loss_, acc_
    
    def load(self, payload: List[torch.Tensor], start_time=None) -> bool:
        self.global_update(payload)
        self.round += 1
        if self.round % self.evaluate_gap == 0:
            loss, acc = self.evaluate(start_time=start_time)
            self.test_loss.append((self.round, loss))
            self.test_acc.append((self.round, acc))
    
class ASManager(ServerManager):
    def __init__(self,
                 network: DistNetwork,
                 handler: ServerHandler,
                 logger: Logger=None,
                 dataset_name: str='mnist'):
        super(ASManager, self).__init__(network, handler)
        self._LOGGER = Logger() if logger is None else logger
        self.message_queue = Queue()

        self.dataset_name = dataset_name

    def main_loop(self):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Raises:
            ValueError: invalid message code.
        """
        updater = threading.Thread(target=self.updater_thread, daemon=True)
        updater.start()

        self.global_start_time = time.time()

        while self._handler.if_stop is not True:
            sender, message_code, payload = self._network.recv()

            if message_code == MessageCode.ParameterRequest:
                self._network.send(content=self._handler.downlink_package,
                                   message_code=MessageCode.ParameterUpdate,
                                   dst=sender)

            elif message_code == MessageCode.ParameterUpdate:
                self.message_queue.put((sender, message_code, payload))

            else:
                raise ValueError(
                    "Unexpected message code {}.".format(message_code))
        
        self._LOGGER.info("Global Training done.")
        self._LOGGER.info("Total time cost: {}s".format(time.time() - self.global_start_time))

    def shutdown(self):
        self.shutdown_clients()
        super().shutdown()

    def updater_thread(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to keep monitoring message queue."""
        while self._handler.if_stop is not True:
            _, message_code, payload = self.message_queue.get()
            self._handler.load(payload, self.global_start_time)

            assert message_code == MessageCode.ParameterUpdate

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.
        """
        for rank in range(1, self._network.world_size):
            self._LOGGER.info("Shutdown client {}.".format(rank))
            _, message_code, _ = self._network.recv(src=rank)  # client request
            if message_code == MessageCode.ParameterUpdate:
                self._network.recv(
                    src=rank
                )  # the next package is model request, which is ignored in shutdown stage.
            self._network.send(message_code=MessageCode.Exit, dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size -
                                                1)
        assert message_code == MessageCode.Exit
    
    
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
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--num_clients', type=int, default=10)

    # 全局轮次与评估间隔
    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--eval_gap', type=int, default=5)

    # 模型与数据集划分
    parser.add_argument('--model', type=str, default="mnist")
    parser.add_argument('--partition', type=str, default="iid", choices=["iid", "noniid-labeldir"])
    parser.add_argument('--dir_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    if args.model == "mnist":
        model = CNN_MNIST()
        if args.partition == "iid":
            dataset_name = "mnist_iid"
            dataset = CompletePartitionedMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients, partition="iid")
        else:
            dataset_name = "mnist_noniid_" + "dir_alpha_" + str(args.dir_alpha)
            dataset = CompletePartitionedMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients, partition="noniid-labeldir", dir_alpha=args.dir_alpha, seed=args.seed)
    else:
        raise NotImplementedError

    logs_path = "../logs/"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    LOGGER = Logger(log_name="async_server", log_file=logs_path + "async_server_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".log")
    
    handler = ASHandler(model, args.global_round, args.num_clients, cuda=torch.cuda.is_available(), logger=LOGGER, evaluate_gap=args.eval_gap)
    handler.setup_optim(args.alpha)
    handler.setup_dataset(dataset)

    # 设置目标精度
    handler.target_accuracy = 0.95

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0, ethernet=args.ethernet)
    
    manager = ASManager(handler=handler, network=network, logger=LOGGER)

    manager.run()
    manager.save_results()




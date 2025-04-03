import argparse
import sys
import torch
from torch import nn
import time
import os
import h5py

sys.path.append("../../")

from typing import List
from copy import deepcopy
from datetime import datetime
from threading import Thread, Event
import queue

from utils.TimeWindowSetting import TimeWindow
from utils.NetworkWrapper import AsyncNetworkWrapper

from fedlab.utils import MessageCode, Logger
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager, ServerManager
from fedlab.core.network import DistNetwork

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.contrib.algorithm.basic_server import ServerHandler
from fedlab.contrib.client_sampler.base_sampler import FedSampler
from fedlab.contrib.client_sampler.uniform_sampler import RandomSampler
from fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from fedlab.models import CNN_CIFAR10, CNN_FEMNIST, CNN_MNIST

POLL_TIMEOUT = 1.0
# Polling timeout for receiving messages

class SemiAsyncServerHandler(ServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int = 0,
        sample_ratio: float = 1,
        cuda: bool = False,
        device: str = None,
        sampler: FedSampler = None,
        logger: Logger = None,
        eval_gap: int = 5,
    ):
        super(SemiAsyncServerHandler, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger
        assert 0.0 <= sample_ratio <= 1.0

        # 基本设置
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.sampler = sampler

        # 客户端训练结果缓存
        self.round_clients = max(
            1, int(self.sample_ratio * self.num_clients)
        )
        self.client_buffer_cache = []

        # 终止条件
        self.global_round = global_round
        self.round = 0

        # 模型评估
        self.eval_gap = eval_gap
        self.test_acc = []
        self.test_loss = []

    @property
    def downlink_package(self):
        return [self.model_parameters, torch.tensor(self.round).to(self.model_parameters.dtype)]

    @property
    def num_clients_per_round(self):
        return self.round_clients

    @property
    def if_stop(self):
        return self.round >= self.global_round

    def sample_clients(self, num_to_sample=None):
        if self.sampler is None:
            self.sampler = RandomSampler(self.num_clients)
        num_to_sample = self.round_clients if num_to_sample is None else num_to_sample
        sampled = self.sampler.sample(self.round_clients)
        self.round_clients = len(sampled)

        assert self.num_clients_per_round == len(sampled)
        return sorted(sampled)

    def global_update(self):
        # 更新全局模型，同时更新相关训练数据;
        # 此外，根据eval_gap定期评估模型
        parameters_list = [ele[0] for ele in self.client_buffer_cache]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        self.round += 1
        self.client_buffer_cache = []

        if self.round > 0 and self.round % self.eval_gap == 0:
            loss, acc = self.evaluate()
            self.test_loss.append((self.round, loss))
            self.test_acc.append((self.round, acc))


    def load(self, payload: List[torch.Tensor]) -> bool:
        # 取出客户端训练结果
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            return True  # 返回True若客户端缓存已满.
        else:
            return False

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

class SemiAsyncServerManager(ServerManager):
    def __init__(
            self,
            network: DistNetwork,
            handler: ServerHandler,
            mode: str = "LOCAL",
            logger: Logger = None,
            tw_setting: TimeWindow = None,
            dataset_name: str = 'mnist'
        ):
        super(SemiAsyncServerManager, self).__init__(network, handler, mode)
        self._LOGGER = Logger() if logger is None else logger
        self.dataset_name = dataset_name

        # 时间窗口
        self.time_window = None
        self.client_time_delay = []
        self.tw_setting = tw_setting

        # 该列表用于记录当前正在进行训练的客户端的rank
        self.busy_clients = set()

        # 该字典用于记录每个客户端的最后激活时间
        self.last_activate_time = {}

    def main_loop(self):
        self.global_start_time = time.time()
        async_network_wrapper = AsyncNetworkWrapper(self._network)

        while self._handler.if_stop is not True:
            # activator = threading.Thread(target=self.activate_clients)
            # activator.start()
            self.activate_clients()

            start_time = time.time()
            # 在时间窗口内等待
            while self.time_window == None or time.time() - start_time < self.time_window:
                sender_rank, message_code, payload = async_network_wrapper.recv_with_timeout(POLL_TIMEOUT)
                if sender_rank is None:
                    continue
                if message_code == MessageCode.ParameterUpdate:
                    # 接收到客户端更新，表示训练结束
                    if sender_rank in self.busy_clients:
                        self.busy_clients.remove(sender_rank)

                    model_version = payload[1].item()
                    if model_version < self._handler.round:
                        # 接收到过时的更新
                        self._LOGGER.info(f"Received outdated updates from client {sender_rank},"
                                          f"Current Model: {self._handler.round} round,"
                                          f"Outdated Updates: {model_version} round.")
                    else:
                        assert self.last_activate_time.get(sender_rank) is not None
                        self.client_time_delay.append(time.time() - self.last_activate_time[sender_rank])
                        if self._handler.load(payload):
                            break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
            
            self._LOGGER.info(f"The {self._handler.round} round: received {len(self.client_time_delay)} updates.")
            
            if len(self.client_time_delay) > 0:
                self._handler.global_update()
                # 设置时间窗口
                if self.tw_setting != None:
                    self.time_window = self.tw_setting(self.client_time_delay)
                    self.client_time_delay = []
                self._LOGGER.info(f"The {self._handler.round - 1} round training done.")
                self._LOGGER.info(f"The time_window of the {self._handler.round} round is {self.time_window}")
            else:
                # 本轮没有收到客户端更新
                self._LOGGER.info(f"No updates received, keep waiting for clients.")
        
        self._LOGGER.info("Global Training done.")
        self._LOGGER.info("Total time cost: {}s".format(time.time() - self.global_start_time))

        unprocessed_messages  = async_network_wrapper.shutdown()
        for (sender_rank, message_code, payload) in unprocessed_messages:
            if message_code == MessageCode.ParameterUpdate:
                self.busy_clients.remove(sender_rank)

        # 等待所有客户端完成训练
        while len(self.busy_clients) > 0:
            sender_rank, message_code, payload = self._network.recv()
            if message_code == MessageCode.ParameterUpdate:
                if sender_rank in self.busy_clients:
                    self.busy_clients.remove(sender_rank)

    def shutdown(self):
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):
        self._LOGGER.info("Client activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client id list: {}".format(clients_this_round))
        # print(rank_dict)

        downlink_package = self._handler.downlink_package
        # self._LOGGER.info(f"Downlink package length: {len(downlink_package)}")
        # for i, tensor in enumerate(downlink_package):
        #     self._LOGGER.info(f"Tensor {i}: shape {tensor.shape}, dtype {tensor.dtype}")

        for rank, values in rank_dict.items():
            # 跳过仍在训练中的客户端
            if rank in self.busy_clients:
                continue

            self._LOGGER.info(f"Preparing data for rank {rank}, client IDs: {values}")
            # Ensure values is not empty
            if not values:
                self._LOGGER.warning(f"No clients mapped to rank {rank}, skipping")
                continue

            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._LOGGER.info(f"ID list shape: {id_list.shape}, dtype: {id_list.dtype}")

            # Prepare send content and validate
            send_content = [id_list] + downlink_package
            # self._LOGGER.info(f"Sending to rank {rank}, content length: {len(send_content)}")
            # for i, item in enumerate(send_content):
            #     if isinstance(item, torch.Tensor):
            #         self._LOGGER.info(f"Item {i}: tensor shape {item.shape}")
            #     else:
            #         self._LOGGER.info(f"Item {i}: {type(item)}")

            # 将参与训练的客户端标记为忙碌
            self.busy_clients.add(rank)

            # 发送数据
            self._network.send(
                content=send_content,
                message_code=MessageCode.ParameterUpdate,
                dst=rank
            )
            # 记录最后激活时间
            self.last_activate_time[rank] = time.time()
            self._LOGGER.info(f"Data sent to rank {rank}")

    def shutdown_clients(self):
        client_list = range(self._handler.num_clients)
        rank_dict = self.coordinator.map_id_list(client_list)

        self._LOGGER.info("Client shutdown procedure.")

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.Exit,
                               dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size -
                                                1)
        assert message_code == MessageCode.Exit

    def save_results(self):
        algo = "_".join(["semi_async", self.dataset_name, str(self._handler.num_clients), str(self._handler.global_round)])

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

    parser = argparse.ArgumentParser(description='Semi-Async Server')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3009')
    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--model', type=str, default="mnist")
    parser.add_argument('--eval_gap', type=int, default=5)
    parser.add_argument('--ethernet', type=str, default=None)
    parser.add_argument('--sample', type=float, default=1)

    args = parser.parse_args()

    if args.model == "mnist":
        model = CNN_MNIST()
        dataset_name = "mnist"
        dataset = PathologicalMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=args.num_clients)
        dataset.preprocess()
    else:
        raise NotImplementedError

    logs_path = "../logs/"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    LOGGER = Logger(log_name="semiasync_server", log_file=logs_path + "semiasync_server_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".log")

    handler = SemiAsyncServerHandler(model=model, global_round=args.global_round, num_clients=args.num_clients, sample_ratio=args.sample, cuda=torch.cuda.is_available(), logger=LOGGER, eval_gap=args.eval_gap)
    handler.setup_dataset(dataset)

    network = DistNetwork(address=(args.ip, args.port),
                        world_size=args.world_size,
                        rank=0,
                        ethernet=args.ethernet)

    manager_ = SemiAsyncServerManager(network=network, handler=handler, logger=LOGGER, tw_setting=TimeWindow.average, dataset_name=dataset_name)

    manager_.run()
    manager_.save_results()
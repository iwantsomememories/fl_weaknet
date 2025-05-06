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
import numpy as np

from utils.TimeWindowScheduler import UCBScheduler, NaiveScheduler, BaseScheduler
from utils.NetworkWrapper import AsyncNetworkWrapper
from utils.Datasets import CompletePartitionedMNIST

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

    def global_update(self, start_time=None):
        # 更新全局模型，同时更新相关训练数据;
        # 此外，根据eval_gap定期评估模型
        parameters_list = [ele[0] for ele in self.client_buffer_cache]
        weights = [ele[2] for ele in self.client_buffer_cache]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        self.round += 1
        self.client_buffer_cache = []

        if self.round > 0 and self.round % self.eval_gap == 0:
            loss, acc = self.evaluate(start_time=start_time)
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

    def setup_dataset(self, dataset: CompletePartitionedMNIST) -> None:
        self.dataset = dataset

    def evaluate(self, start_time):
        self._model.eval()
        test_loader = self.dataset.get_dataloader(batch_size=128, type="test")
        loss_, acc_ = evaluate(self._model, nn.CrossEntropyLoss(), test_loader)
        cur_time = time.time() - start_time
        self._LOGGER.info(
            f"Round [{self.round}/{self.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}% Curtime: {cur_time:.2f}s"
        )

        return loss_, acc_

class SemiAsyncServerManager(ServerManager):
    def __init__(
            self,
            network: DistNetwork,
            handler: ServerHandler,
            mode: str = "LOCAL",
            logger: Logger = None,
            time_scheduler: BaseScheduler = None,
            dataset_name: str = 'mnist'
        ):
        super(SemiAsyncServerManager, self).__init__(network, handler, mode)
        self._LOGGER = Logger() if logger is None else logger
        self.dataset_name = dataset_name

        # 时间窗口
        self.time_window = None
        self.client_time_delay = []
        self.time_scheduler = time_scheduler

        # 该列表用于记录当前正在进行训练的客户端的rank
        self.busy_clients = set()

        # 该字典用于记录每个客户端的最后激活时间
        self.last_activate_time = {}
    
    def pre_train(self, round: int = 5):
        self._LOGGER.info("Server pre-train procedure is running")
        hist_client_delay = {}
        average_client_delay = []

        for i in range(self._handler.num_clients):
            hist_client_delay[i] = []        
        
        for ep in range(round):
            self.activate_clients()

            count = 0
            while count < self._handler.num_clients:
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    # 接收到客户端更新，表示训练结束
                    if sender_rank in self.busy_clients:
                        self.busy_clients.remove(sender_rank)
                    assert self.last_activate_time.get(sender_rank) is not None
                    hist_client_delay[sender_rank - 1].append(time.time() - self.last_activate_time[sender_rank])
                    count += 1
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
        
        self._LOGGER.info("Server pre-train procedure is finished")

        for delays in hist_client_delay.values():
            average_client_delay.append(sum(delays)/len(delays))
        
        q25 = np.percentile(average_client_delay, 25)
        median = np.median(average_client_delay)
        q75 = np.percentile(average_client_delay, 75)
        max_value = np.max(average_client_delay)
        candidate_windows = [q25, median, q75, max_value]
        self.candidate_windows = sorted(candidate_windows)
        self._LOGGER.info(f"Candidate time windows: {candidate_windows}")   

    def main_loop(self):
        # 预训练阶段
        self.pre_train(round=3)
        if isinstance(self.time_scheduler, UCBScheduler):
            self.time_scheduler.set_candidate_windows(self.candidate_windows)
            self.time_window = self.time_scheduler.select_window()
        elif isinstance(self.time_scheduler, NaiveScheduler):
            self.time_window = self.candidate_windows[1] # 选择中位数作为初始时间窗口
        else:
            raise NotImplementedError("Invalid time scheduler")

        self.busy_clients.clear()
        self.global_start_time = time.time()
        async_network_wrapper = AsyncNetworkWrapper(self._network)

        while self._handler.if_stop is not True:
            # activator = threading.Thread(target=self.activate_clients)
            # activator.start()
            self.activate_clients()

            self._LOGGER.info(f"Round: {self._handler.round}, time_window: {self.time_window}")
            start_time = time.time()
            # 在时间窗口内等待
            while time.time() - start_time < self.time_window:
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
                        self._LOGGER.info(f"Received updates from client {sender_rank}.")
                        if self._handler.load(payload):
                            break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
            
            self._LOGGER.info(f"The {self._handler.round} round: received {len(self.client_time_delay)} updates.")
            

            if isinstance(self.time_scheduler, UCBScheduler):
                if len(self.client_time_delay) > 0:
                    client_params = [ele[0] for ele in deepcopy(self._handler.client_buffer_cache)]
                    reward = self.time_scheduler.get_reward(global_params=self._handler.model_parameters, client_params_list=client_params, time_window=self.time_window)
                else:
                    reward = 0
                self.time_scheduler.update(self.time_window, reward)
                self.time_window = self.time_scheduler.select_window()
            elif isinstance(self.time_scheduler, NaiveScheduler):
                self.time_window = self.time_scheduler.select_window(self.client_time_delay, self.time_window)
            else:
                raise NotImplementedError("Invalid time scheduler")

            if len(self.client_time_delay) > 0:
                self._handler.global_update(self.global_start_time)
            else:
                # 本轮没有收到客户端更新
                self._handler.round += 1
            
            self.client_time_delay = []
        
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
        
        self._LOGGER.info("All clients have finished training.")

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

            # Prepare send content and validate
            send_content = downlink_package
            self._LOGGER.info(f"Sending to rank {rank}, content length: {len(send_content)}")

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
            self._network.send(content=downlink_package,
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
    parser.add_argument('--ethernet', type=str, default=None)

    parser.add_argument('--world_size', type=int, default=11)
    parser.add_argument('--num_clients', type=int, default=10)

    parser.add_argument('--global_round', type=int, default=100)
    parser.add_argument('--eval_gap', type=int, default=5)

    parser.add_argument('--model', type=str, default="mnist")
    parser.add_argument('--partition', type=str, default="iid", choices=["iid", "noniid-labeldir"])
    parser.add_argument('--dir_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--sample', type=float, default=1)

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
    LOGGER = Logger(log_name="semiasync_server", log_file=logs_path + "semiasync_server_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".log")

    handler = SemiAsyncServerHandler(model=model, global_round=args.global_round, num_clients=args.num_clients, sample_ratio=args.sample, cuda=torch.cuda.is_available(), logger=LOGGER, eval_gap=args.eval_gap)
    handler.setup_dataset(dataset)

    network = DistNetwork(address=(args.ip, args.port),
                        world_size=args.world_size,
                        rank=0,
                        ethernet=args.ethernet)
    
    time_scheduler = UCBScheduler(args.num_clients, 1.0, 1.0)
    # time_scheduler = NaiveScheduler("fixed")

    manager_ = SemiAsyncServerManager(network=network, handler=handler, logger=LOGGER, time_scheduler=time_scheduler, dataset_name=dataset_name)

    manager_.run()
    manager_.save_results()
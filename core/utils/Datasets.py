import os
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from fedlab.contrib.dataset import PathologicalMNIST, PartitionedMNIST, PartitionedCIFAR10, BaseDataset
from fedlab.utils.dataset.partition import MNISTPartitioner
from fedlab.contrib.dataset.basic_dataset import FedDataset, Subset
from fedlab.models import CNN_MNIST

class CompletePartitionedMNIST(PartitionedMNIST):
    def __init__(self,
                 root,
                 path,
                 num_clients,
                 download=True,
                 preprocess=False,
                 partition="iid",
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        
        # 设置默认transform（只包含必要的转换）
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transform
            
        self.target_transform = target_transform

        if preprocess:
            self.preprocess(partition=partition,
                          dir_alpha=dir_alpha,
                          verbose=verbose,
                          seed=seed,
                          download=download)

    def preprocess(self,
                  partition="iid",
                  dir_alpha=None,
                  verbose=True,
                  seed=None,
                  download=True):
        
        if os.path.exists(self.path) is not True:
            os.makedirs(self.path, exist_ok=True)
        if os.path.exists(os.path.join(self.path, "train")) is not True:
            os.makedirs(os.path.join(self.path, "train"), exist_ok=True)
        if os.path.exists(os.path.join(self.path, "test")) is not True:
            os.makedirs(os.path.join(self.path, "test"), exist_ok=True)
        
        # 加载原始数据集时不应用transform，我们稍后手动处理
        trainset = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=download,
            transform=None  # 重要：这里设置为None
        )

        partitioner = MNISTPartitioner(
            trainset.targets,
            self.num_clients,
            partition=partition,
            dir_alpha=dir_alpha,
            verbose=verbose,
            seed=seed
        )

        # 预处理并保存训练子集
        subsets = {}
        for cid in range(self.num_clients):
            # 手动应用transform到每个样本
            indices = partitioner.client_dict[cid]
            data = []
            targets = []
            for idx in indices:
                img, target = trainset[idx]
                if self.transform is not None:
                    img = self.transform(img)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                data.append(img)
                targets.append(target)
            
            # 创建已转换的数据集子集
            subset = BaseDataset(data, targets)
            subsets[cid] = subset
            torch.save(
                subset,
                os.path.join(self.path, "train", f"data{cid}.pkl")
            )

        # 预处理测试集
        testset = torchvision.datasets.MNIST(
            root=self.root,
            train=False,
            download=download,
            transform=None
        )
        
        test_data = []
        test_targets = []
        for img, target in testset:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            test_data.append(img)
            test_targets.append(target)
        
        test_dataset = BaseDataset(test_data, test_targets)
        torch.save(test_dataset, os.path.join(self.path, "test", "test.pkl"))

    def get_dataset(self, cid=None, type="train"):
        if type == "test":
            dataset = torch.load(os.path.join(self.path, "test", "test.pkl"))
        elif type == "train":
            dataset = torch.load(os.path.join(self.path, "train", f"data{cid}.pkl"))
        else:
            raise ValueError(f"Unknown dataset type: {type}")
        
        return dataset

    def get_dataloader(self, cid=None, batch_size=None, type="train"):    
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


def count_classes_loader(dataset, batch_size=256):
    counter = Counter()
    loader = DataLoader(dataset, batch_size=batch_size)
    for _, labels in loader:
        counter.update(labels.tolist())

    return sorted(counter.items(), key=lambda x: x[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-async Client")

    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--partition', type=str, default="iid", 
                      choices=["iid", "noniid-labeldir"])
    parser.add_argument('--dir_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if args.dataset == "mnist":
        dataset = CompletePartitionedMNIST(
            root='../datasets/mnist/',
            path="../datasets/mnist/",
            num_clients=args.num_clients,
            preprocess=True,
            transform=transforms.Compose([  # 明确指定transform流程
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            partition=args.partition,
            dir_alpha=args.dir_alpha,
            seed=args.seed,
        )

        # 查看数据分布
        for i in range(0, args.num_clients):
            train = dataset.get_dataset(i, "train")
            print("data size of client {}: {}".format(i, len(train)))
            print("data distribution: {}".format(count_classes_loader(train)))

        test = dataset.get_dataset(type="test")
        print("data size of test: {}".format(len(test)))
        print("data distribution: {}".format(count_classes_loader(test)))

        # 检查数据格式是否正确
        # model = CNN_MNIST()
        # train_loader = dataset.get_dataloader(cid=0, batch_size=32, type="train")

        # model.train()
        # for data, target in train_loader:
        #     output = model(data)

    else:
        raise NotImplementedError
    

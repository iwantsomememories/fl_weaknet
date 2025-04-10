
from fedlab.contrib.dataset import PathologicalMNIST, PartitionedMNIST, PartitionedCIFAR10
import sys

sys.path.append("../../")

# dataset = PathologicalMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=10)
# dataset.preprocess()

seed = 0

dataset = PartitionedMNIST(root='../datasets/mnist/', path="../datasets/mnist/", num_clients=5, partition="noniid-labeldir", dir_alpha=0.5, seed=seed)

dataset.preprocess()

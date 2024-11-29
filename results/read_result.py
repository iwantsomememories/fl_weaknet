from h5py import File, Dataset, Group
from matplotlib import pyplot as plt
import numpy as np

def getacc (f):
    for key in f.keys():
        if isinstance(f[key], Dataset) and f[key].name == "/test_acc":
            round = [x[0] for x in f[key]]
            acc = [x[1] for x in f[key]]
            return round, acc

def getloss(f):
    for key in f.keys():
        if isinstance(f[key], Dataset) and f[key].name == "/test_loss":
            round = [x[0] for x in f[key]]
            loss = [x[1] for x in f[key]]
            return round, loss

def draw_acc(name):
    f = File("async_mnist_5_20.h5", "r")

    round, acc = getacc(f)

    f.close()

    plt.plot(round, acc, marker=".")


    # xticks = [i for i in range(0, 51, 5)]
    # yticks = np.arange(0, 1.05, 0.05)

    # plt.xticks(xticks)
    # plt.yticks(yticks)

    plt.xlabel("round")
    plt.ylabel("acc")

    # plt.legend(fontsize=15)

    plt.title("Averaged accuracy under varying timeout rates in iid setting", fontsize=20)

    plt.savefig(name)


def draw_loss(name):
    f0 = File("mnist_FedAvg_test_droprate=0_noniid_0.h5", "r")
    f25 = File("mnist_FedAvg_test_droprate=0.25_noniid_0.h5", "r")
    f50 = File("mnist_FedAvg_test_droprate=0.50_noniid_0.h5", "r")
    f75 = File("mnist_FedAvg_test_droprate=0.75_noniid_0.h5", "r")

    loss0 = getloss(f0)
    loss25 = getloss(f25)
    loss50 = getloss(f50)
    loss75 = getloss(f75)

    f0.close()
    f25.close()
    f50.close()
    f75.close()

    round = [i for i in range(0, 51, 5)]

    plt.plot(round, loss0, label="timeout rate = 0", marker=".")
    plt.plot(round, loss25, label="timeout rate = 0.25", marker=".")
    plt.plot(round, loss50, label="timeout rate = 0.50", marker=".")
    plt.plot(round, loss75, label="timeout rate = 0.75", marker=".")

    # xticks = [i for i in range(0, 51, 5)]
    # yticks = np.arange(0, 1.05, 0.05)

    # plt.xticks(xticks)
    # plt.yticks(yticks)

    plt.xlabel("round", fontsize=15)
    plt.ylabel("loss", fontsize=15)

    plt.legend(fontsize=15)

    plt.title("Averaged loss under varying timeout rates in iid setting", fontsize=20)

    plt.savefig(name)


if __name__ == "__main__":
    draw_acc("mnist.png")

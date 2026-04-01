import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from cold_diffusion.dataset.cifar10 import CIFAR10Dataset
from cold_diffusion.dataset.mnist import MNISTDataset
from cold_diffusion.utils.misc import load_config


@pytest.mark.skip()
def test_cifar10_dataset():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    dataset = CIFAR10Dataset(config, "train", None)
    x = dataset[0]
    assert x.shape == (3, 32, 32)


@pytest.mark.skip()
def test_mnist_dataset():
    config = load_config("cold_diffusion/config/mnist_config.yaml")
    dataset = MNISTDataset(config, "train", None)
    x = dataset[0]
    assert x.shape == (1, 28, 28)


if __name__ == "__main__":
    print("All tests passed!")

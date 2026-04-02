from cold_diffusion.dataset.animal_mnist import AnimalMNISTDataset
from cold_diffusion.dataset.celeba import CelebADataset
from cold_diffusion.dataset.cifar10 import CIFAR10Dataset
from cold_diffusion.dataset.fashion_mnist import FashionMNISTDataset
from cold_diffusion.dataset.flowers102 import Flowers102Dataset
from cold_diffusion.dataset.mnist import MNISTDataset

__all_datasets__ = {
    "CIFAR10Dataset": CIFAR10Dataset,
    "MNISTDataset": MNISTDataset,
    "CelebADataset": CelebADataset,
    "FashionMNISTDataset": FashionMNISTDataset,
    "AnimalMNISTDataset": AnimalMNISTDataset,
    "Flowers102Dataset": Flowers102Dataset,
}

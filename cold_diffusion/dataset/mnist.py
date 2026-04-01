import torchvision.transforms.v2 as T
from torchvision.datasets import MNIST

from cold_diffusion.dataset.dataset_base import DatasetBaseClass


class MNISTDataset(DatasetBaseClass):
    def __init__(self, cfg, mode, logger):
        super().__init__(cfg, mode, logger)
        self.transform = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
        self.dataset = MNIST(
            root=self.root_dir,
            train=self.train,
            download=True,
        )
        self.get_channel_stats()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img

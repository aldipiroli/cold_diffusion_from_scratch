import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.logger = logger

        self.root_dir = cfg["DATA"]["root_dir"]
        self.train = mode == "train"

        self.dataset = CIFAR10(
            root=self.root_dir,
            train=self.train,
            download=True,
        )
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img

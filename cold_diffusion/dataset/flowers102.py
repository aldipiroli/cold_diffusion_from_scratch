import torchvision.transforms.v2 as T
from torchvision.datasets import Flowers102

from cold_diffusion.dataset.dataset_base import DatasetBaseClass


class Flowers102Dataset(DatasetBaseClass):
    def __init__(self, cfg, mode, logger):
        super().__init__(cfg, mode, logger)
        img_size = cfg["DATA"]["img_size"]
        self.transform = T.Compose(
            [
                T.Resize((img_size[1], img_size[2])),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.dataset = Flowers102(
            root=self.root_dir,
            split="train" if self.train else "test",
            download=True,
        )
        self.get_channel_stats()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img

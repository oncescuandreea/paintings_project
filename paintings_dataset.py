from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from zsvision.zs_beartype import beartype


class PaintingsDataset(Dataset):
    @beartype
    def __init__(
            self,
            data_dir: Path,
            csv_path: Path,
            split: str,
            im_suffix: str,
            # writer: torch.utils.tensorboard.writer.SummaryWriter,
            transform: transforms.Compose = None,
    ):

        with open(csv_path / f"artuk_{split}.csv", "r") as f:
            csv_contents = pd.read_csv(f)
            self.painter_names = list(csv_contents.iloc[:, 0])
            self.paintings = list(csv_contents.iloc[:, 1])
        self.data_dir = data_dir
        self.im_suffix = im_suffix
        self.transform = transform
        self.label_dict = self.label2idx(self.painter_names)
        idx_labels = []
        for el in self.painter_names:
            idx_labels.append(self.label_dict[el])
        self.labels = idx_labels

    @beartype
    def label2idx(self, labels: list) -> Dict:
        top_labels = Counter(labels).most_common()
        label_dict = {}
        for i, key in enumerate(top_labels):
            label_dict[key[0]] = i

        return label_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        picture_name = self.data_dir / f"{self.paintings[idx]}{self.im_suffix}"
        img = Image.open(picture_name)
        transform_no_change = transforms.Compose([transforms.CenterCrop(69),
                                                  transforms.Resize(224),
                                                  transforms.ToTensor()])
        if self.transform is not None:
            img_org = transform_no_change(img)
            img = self.transform(img)

        return img_org, img, label

from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
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
            csv_file_path: Path,
            split: str,
            im_suffix: str,
            keep_k_most_common_labels: int,
            # writer: torch.utils.tensorboard.writer.SummaryWriter,
            transform: transforms.Compose = None,
    ):
        with open(csv_file_path, "r") as f:
            csv_contents = pd.read_csv(f)
        self.csv_contents = csv_contents
        self.data_dir = data_dir
        self.csv_contents = self.sanitise_csv_contents(self.csv_contents)
        im_info = self.filter_to_k_most_common_labels(k=keep_k_most_common_labels)
        self.label2idx = im_info["label_dict"]
        keep = np.where(np.array(im_info["splits"]) == split)[0]
        # keep = np.array([0] * len(keep))
        self.labels = np.array(im_info["labels"])[keep]
        self.images = [data_dir / im_name for im_name
                       in np.array(im_info["images"])[keep]]
        self.transform = transform

    @beartype
    def sanitise_csv_contents(self, csv_contents: pd.DataFrame) -> pd.DataFrame:
        """This function fixes inconsistencies in the image naming convention used
        to store the paintings.  Some images have the form:
            {name_and_date}-{number}
        whereas others have the form:
            {name_and_date}_{number}
        To address this, we loop over each image file and check each naming style for
        existence, then update the meta data accordingly.

        Args:
            csv_contents: the contents of the csv provided by Attila.

        Returns:
            the contents of the csv after fixing the image paths.
        """
        sanitised = []
        img_files = list(csv_contents.img_file)
        # These files do not exist in our image collection
        exclude = {
            "rwa-spa_2016-227.jpg",
            "rwa-spa_2016-238.jpg",
            "rwa-spa_2016-239.jpg",
            "rwa-spa_2016-240.jpg",
            "rwa-spa_2018-178.jpg",
        }
        keep = []
        for ii, img_file in enumerate(img_files):
            if img_file in exclude:
                continue
            elif (self.data_dir / img_file).exists():
                pass
            else:
                # replace the last occurence of a hyphen with underscore to match
                # inconsistent filenames
                img_file = img_file[::-1].replace("-", "_", 1)[::-1]
                msg = f"img_file: {img_file} does not exist!"
                assert (self.data_dir / img_file).exists(), msg
            keep.append(ii)
            sanitised.append(img_file)
        keep = np.array(keep)
        csv_contents = csv_contents.iloc[csv_contents.index.isin(keep)]
        pd.options.mode.chained_assignment = None
        csv_contents.loc[:, "img_file"] = sanitised
        return csv_contents

    @beartype
    def filter_to_k_most_common_labels(self, k: int) -> Dict:
        labels = list(self.csv_contents['medium'])
        topk_classes = Counter(labels).most_common(k)
        min_number = int(topk_classes[k - 1][1])
        topk_classes = sorted(topk_classes)
        label_dict = {}
        for i, el in enumerate(topk_classes):
            label_dict[el[0]] = i
        images = []
        labels = []
        for i, el in enumerate(self.csv_contents['img_file']):
            if self.csv_contents.iloc[i]['medium'] in dict(topk_classes).keys():
                images.append(el)
                labels.append(label_dict[self.csv_contents.iloc[i]['medium']])
        labels = np.array(labels)
        images = np.array(images)
        balanced_labels, balanced_ims = [], []
        num_train = int(min_number * 0.8)
        splits = []
        for subset in ("train", "val"):
            for class_idx in label_dict.values():
                keep = np.where(labels == class_idx)[0][:min_number]
                if subset == "train":
                    keep = keep[:num_train]
                elif subset == "val":
                    keep = keep[num_train:]
                else:
                    raise ValueError(f"Unknown subset: {subset}")
                balanced_ims.extend(images[keep])
                balanced_labels.extend(labels[keep])
                splits.extend([subset] * keep.size)
        im_info = {
            "label_dict": label_dict,
            "images": balanced_ims,
            "labels": balanced_labels,
            "splits": splits,
        }
        return im_info

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.images[idx])
        transform_no_change = transforms.Compose([transforms.CenterCrop(500),
                                                  transforms.Resize(224),
                                                  transforms.ToTensor()])
        if self.transform is not None:
            img_org = transform_no_change(img)
            img = self.transform(img)

        return img_org, img, label

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


class paintingsDataset(Dataset):
    def __init__(self, data_dir, csv_file_path, split, transform=None):
        """
        Args:
            csv_file_path: Path to the csv file with document names, gender and other info
            root_dir: Directory with all paintings
            transform: Optional transform to be applied on a sample
        """
        with open(csv_file_path, "r") as f:
            csv_contents = pd.read_csv(f)
        self.csv_contents = csv_contents
        self.data_dir = data_dir
        images = []
        labels = []
        pattern = r'(\w+)-(\w+)_(\d+)'
        number_pattern = r'(\d+)'
        for i, img in enumerate(csv_contents['img_file']):
            name_and_date = re.match(pattern, img)[0]
            number = re.findall(number_pattern, img)[1]
            if f"{name_and_date}-{number}.jpg" in os.listdir(self.data_dir):
                images.append(f"{self.data_dir}/{img}")
                labels.append(csv_contents['gender'][i])
            else:
                if f"{name_and_date}_{number}.jpg" in os.listdir(self.data_dir):
                    images.append(f"{self.data_dir}/{name_and_date}_{number}.jpg")
                    labels.append(csv_contents['gender'][i])
        train_size = int(np.floor(len(labels)*0.6))
        val_size = int(np.floor(len(labels)*0.2))
        if split == 'train':
            self.labels = labels[0:train_size]
            self.images = images[0:train_size]
        elif split == 'val':
            self.labels = labels[train_size:train_size+val_size]
            self.images = images[train_size:train_size+val_size]
        else:
            self.labels = labels[train_size+val_size:]
            self.images = images[train_size+val_size:]

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        label = self.labels[idx]
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


paintings_train = paintingsDataset(
    data_dir="/users/oncescu/data/pictures_project/pictures/spa_images",
    csv_file_path="/users/oncescu/data/pictures_project/spa-classify.csv?dl=0",
    split='train')
paintings_val = paintingsDataset(
    data_dir="/users/oncescu/data/pictures_project/pictures/spa_images",
    csv_file_path="/users/oncescu/data/pictures_project/spa-classify.csv?dl=0",
    split='val')
paintings_test= paintingsDataset(
    data_dir="/users/oncescu/data/pictures_project/pictures/spa_images",
    csv_file_path="/users/oncescu/data/pictures_project/spa-classify.csv?dl=0",
    split='test')

fig = plt.figure()

# for i in range(len(paintings_dataset)):
#    img, label = paintings_dataset[i]
#    print(i, img.size, label)

# test_set = paintings_dataset[876:]
# train_set, val_set = torch.utils.data.random_split(paintings_dataset,
#                 [657, len(paintings_dataset) - len(test_set)
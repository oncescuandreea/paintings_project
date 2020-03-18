import argparse
import collections
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from main import PaintingsDataset
from zsvision.zs_utils import BlockTimer


# class PaintingsDataset(Dataset):
#     def __init__(self, data_dir, csv_file_path):
#         """
#         Args:
#             csv_file_path: Path to the csv file with document names, medium and other info
#             root_dir: Directory with all paintings
#             transform: Optional transform to be applied on a sample
#         """
#         with open(csv_file_path, "r") as f:
#             csv_contents = pd.read_csv(f)
#         self.csv_contents = csv_contents
#         self.data_dir = data_dir
#         images = []
#         labels = []
#         pattern = r'(\w+)-(\w+)_(\d+)'
#         number_pattern = r'(\d+)'
#         list_of_images_folder = os.listdir(self.data_dir)
#         dict_of_classes, min_examples = top5(csv_file_path)
#         counter_classes = dict(zip(list(dict_of_classes.keys()), [0]*len(dict_of_classes)))
#         for i, img in enumerate(csv_contents['img_file']):
#             if (csv_contents['medium'][i] in dict_of_classes and
#                     counter_classes[csv_contents['medium'][i]] < min_examples):
#                 name_and_date = re.match(pattern, img)[0]
#                 number = re.findall(number_pattern, img)[1]
#                 if f"{name_and_date}-{number}.jpg" in list_of_images_folder:
#                     images.append(f"{self.data_dir}/{img}")
#                     labels.append(dict_of_classes[csv_contents['medium'][i]])
#                     counter_classes[csv_contents['medium'][i]] += 1
#                 else:
#                     if f"{name_and_date}_{number}.jpg" in list_of_images_folder:
#                         images.append(f"{self.data_dir}/{name_and_date}_{number}.jpg")
#                         labels.append(dict_of_classes[csv_contents['medium'][i]])
#                         counter_classes[csv_contents['medium'][i]] += 1
    
#         mean, var = meanvar(images)
#         print(f"mean: {mean}")
#         print(f"var: {var}")

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # if torch.is_tensor(idx):
#         #     idx = idx.tolist()
#         label = self.labels[idx]
#         img = Image.open(self.images[idx])
#         if self.transform is not None:
#             img = self.transform(img)
#         # img = np.array(img)
#         return img, label


def top5(csv_file_path):
    labels = []
    with open(csv_file_path, "r") as f:
        csv_contents = pd.read_csv(f)
    labels = list(csv_contents['medium'])
    top5_classes = Counter(labels).most_common(5)
    min_number = int(top5_classes[4][1])
    top5_classes = sorted(top5_classes)
    label_dict = {}
    for i, el in enumerate(top5_classes):
        label_dict[el[0]] = i
    return label_dict, min_number


def meanvar(images):
    sumrgb = 3 * [0]
    lenrgb = 3 * [0]
    for image in images:
        img = Image.open(image)
        transform_tensor = transforms.ToTensor()
        img = transform_tensor(img)
        for i in range(3):
            sumrgb[i] += img[i].sum()
            lenrgb[i] += img[i].size()[0] * img[i].size()[1]
    meanrgb = np.divide(sumrgb, lenrgb)
    sumrgb = 3 * [0]
    print("calculated mean")
    for k, image in enumerate(images):
        img = Image.open(image)
        transform_tensor = transforms.ToTensor()
        img = transform_tensor(img)
        for i in range(3):
            for row in img[i]:
                sumrgb[i] += sum([(el.item() - meanrgb[i])**2 for el in row])
        print(k)
    varrgb = np.divide(sumrgb, lenrgb)
    return meanrgb, varrgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default="data/spa-classify.csv?dl=0", type=Path)
    parser.add_argument('--im_dir', default="data/pictures/spa_images", type=Path)
    parser.add_argument('--ckpt_path', default="data/model.pt", type=Path)
    args = parser.parse_args()

    with BlockTimer("computing statistics for dataset"):
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor()])
        dataset = PaintingsDataset(
            data_dir=args.im_dir,
            split="train",
            transform=transform,
            csv_file_path=args.csv_path,
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        for minibatch in loader:
            ims, _ = minibatch
            ims = ims.refine_names("N", "C", "H", "W")
            rgb_mean = ims.mean(("N", "H", "W"))
            rgb_std = ims.std(("N", "H", "W"))


if __name__ == '__main__':
    main()

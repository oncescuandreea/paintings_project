import argparse
import itertools
import os
import re
import subprocess
from collections import Counter, defaultdict
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
from zsvision.zs_beartype import beartype
from typing import Dict, Tuple, List
from zsvision.zs_utils import BlockTimer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PaintingsDataset(Dataset):
    @beartype
    def __init__(
            self,
            data_dir: Path,
            csv_file_path: Path,
            split: str,
            im_suffix: str,
            keep_k_most_common_labels: int,
            transform: transforms.Compose = None,
    ):
        with open(csv_file_path, "r") as f:
            csv_contents = pd.read_csv(f)
        self.csv_contents = csv_contents
        self.data_dir = data_dir
        self.csv_contents = self.sanitise_csv_contents(self.csv_contents)

        images = []
        labels = []

        list_of_images = list(data_dir.glob(f"*{im_suffix}"))
        print(f"Found {len(list_of_images)} in {data_dir}")
        # list_of_images_folder = os.listdir(self.data_dir)
        label2idx, min_examples_per_label = self.filter_to_k_most_common_labels(
            k=keep_k_most_common_labels)

        # dict_of_classes, min_examples = top5(csv_file_path)

        train_size = int(np.floor(min_examples_per_label*0.8))
        val_size = int(np.floor(min_examples_per_label*0.2))
        dict_of_labels = stratify(labels)
        if split == 'train':
            self.labels = list(itertools.chain(*([list(map(labels.__getitem__,
                                                           dict_of_labels[i][0:train_size]))
                                                  for i in range(5)]*7)))
            self.images = list(itertools.chain(*([list(map(images.__getitem__,
                                                           dict_of_labels[i][0:train_size]))
                                                  for i in range(5)]*7)))
        elif split == 'val':
            self.labels = list(itertools.chain(*[list(
                map(labels.__getitem__, dict_of_labels[i][train_size:train_size+val_size]))
                                                 for i in range(5)]))
            self.images = list(itertools.chain(*[list(map(images.__getitem__, dict_of_labels[i][train_size:train_size+val_size])) for i in range(5)]))
        else:
            self.labels = list(itertools.chain(*[list(map(labels.__getitem__, dict_of_labels[i][train_size+val_size:])) for i in range(5)]))
            self.images = list(itertools.chain(*[list(map(images.__getitem__, dict_of_labels[i][train_size+val_size:])) for i in range(5)]))

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
    def filter_to_k_most_common_labels(self, k: int) -> Tuple[Dict, int]:
        labels = list(self.csv_contents['medium'])
        topk_classes = Counter(labels).most_common(k)
        min_number = int(topk_classes[k - 1][1])
        topk_classes = sorted(topk_classes)
        label_dict = {}
        for i, el in enumerate(topk_classes):
            label_dict[el[0]] = i
        return label_dict, min_number

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        label = self.labels[idx]
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        # img = np.array(img)
        return img, label

def stratify(labels):
    dict_of_lists = defaultdict(list)
    for i, el in enumerate(labels):
        dict_of_lists[el].append(i)
    return dict_of_lists



def imshow(img):
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    img -= img.min()
    img /= img.max()
    plt.imshow(img)
    plt.show()
    fig_dir = Path("data/figsand")
    fig_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(fig_dir / "im.png"))


@beartype
def train(
        ckpt_path: Path,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    plt.savefig('test.jpg')

    net = Net()
    if epoch != 0:
        net = torch.load(ckpt_path)
        net.eval()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        predicted = outputs.argmax(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss:%.3f' %
                  (epoch + 1, i + 1, running_loss/100))
            running_loss = 0.0
    print(f"accuracy on train data after one epoch is:{correct / total}")
    torch.save(net, ckpt_path)
    print('Finished Training')
    return 100 * correct / total


@beartype
def testval(
        ckpt_path: Path,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        visualise: Path
):
    net = Net()
    net = torch.load(ckpt_path)
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    # bash_command = f"rm {visualise}"
    # subprocess.call(bash_command.split())
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            total += labels.size(0)
            # correct += (outputs.argmax(1) == labels).sum().item()
            print(f"outputs: {outputs}")
            print(f"labels: {labels}")
            predicted = outputs.argmax(1)
            print(f"predicted: {predicted}")
            correct += (predicted == labels).sum().item()
            print('correct: %d' % correct)
            print('total: %d ' % total)
            print('\n')
            inputs = inputs.cpu()
            labels = labels.cpu()
            predicted = predicted.cpu()
            imshow(torchvision.utils.make_grid(inputs))
            plt.savefig(f"data/visualise/{total/4}_{('-').join(list(map(str, labels.tolist())))}_{('-').join(list(map(str, predicted.tolist())))}.jpg")
    acc = 100 * correct / total
    print(f"Accuracy of the network on the validation set is: {acc} %%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default="data/spa-classify.csv?dl=0", type=Path)
    parser.add_argument('--im_dir', default="data/pictures/spa_images", type=Path)
    parser.add_argument('--ckpt_path', default="data/model.pt", type=Path)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--keep_k_most_common_labels', default=5, type=int)
    parser.add_argument('--dataset', default="five-class")
    parser.add_argument('--im_suffix', default=".jpg",
                        help="the suffix for images in the dataset")
    parser.add_argument('--visualise', default="data/visualise/*", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == "five-class":
        mean = (0.5827, 0.5678, 0.5294)
        std = (0.2707, 0.2595, 0.2776)
    else:
        raise NotImplementedError(f"Means and std are not computed for {args.dataset}")

    transform_train = transforms.Compose([transforms.RandomAffine(degrees=90),
                                          transforms.ColorJitter(contrast=0.2),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_val = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    dataset_kwargs = {
        "data_dir": args.im_dir,
        "csv_file_path": args.csv_path,
        "keep_k_most_common_labels": args.keep_k_most_common_labels,
        "im_suffix": args.im_suffix,
    }
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }
    paintings_train = PaintingsDataset(
        split='train',
        transform=transform_train,
        **dataset_kwargs,
    )
    # imag, _ = paintings_train[0]
    # print(imag.size())
    train_loader = torch.utils.data.DataLoader(
        paintings_train,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    paintings_val = PaintingsDataset(
        split="val",
        transform=transform_val,
        **dataset_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=paintings_val,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    accuracytrain = []
    label_dict, _ = top5(args.csv_path)
    for epoch in range(args.num_epochs):
        with BlockTimer(f"[{epoch}/{args.num_epochs} training and eval"):
            acc = train(
                epoch=epoch,
                device=device,
                ckpt_path=args.ckpt_path,
                train_loader=train_loader,
            )
            accuracytrain.append(acc)
            testval(ckpt_path=args.ckpt_path, val_loader=val_loader, device=device,
                    visualise=args.visualise)
    print(accuracytrain)
    print(label_dict)

if __name__ == '__main__':
    main()

import argparse
import glob
import time
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from zsvision.zs_beartype import beartype
from zsvision.zs_utils import BlockTimer

from metrics import AverageMeter, ProgressMeter, accuracy


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
        im_info = self.filter_to_k_most_common_labels(k=keep_k_most_common_labels)
        self.label2idx = im_info["label_dict"]
        keep = np.where(np.array(im_info["splits"]) == split)[0]
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
def build_summary_writer(learning_rate: float, batch_size: int) -> SummaryWriter:
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    log_dir = Path("data/logs") / f"lr-{learning_rate}-bs-{batch_size}" / timestamp
    log_dir.mkdir(exist_ok=True, parents=True)
    return SummaryWriter(str(log_dir))


@beartype
def train(
        ckpt_path: Path,
        net: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
        learning_rate: float,
        frequency: int,
):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    plt.savefig('test.jpg')
    net.train()
    # if epoch != 0:
    #     net = torch.load(ckpt_path)
    #     net.eval()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    running_loss = 0.0
    running_loss_tot = 0.0
    total = 0
    correct = 0

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data[0].to(device), data[1].to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        outputs = net(inputs)

        # predicted = outputs.argmax(1)
        # correct += (predicted == labels).sum().item()
        # total += labels.size(0)

        acc1 = accuracy(output=outputs, target=labels, topk=(1,))

        loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), images.size(0))

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        running_loss += loss.item()
        running_loss_tot += loss.item()
        if i % frequency == 0:
            progress.display(i)

        # if i % 10 == 9:
        #     print('[%d, %5d] loss:%.3f' %
        #           (epoch + 1, i + 1, running_loss/100))
        #     running_loss = 0.0
    # print(f"accuracy on train data after one epoch is:{correct / total}")
    print(f" * Acc@1 {top1.avg:.3f}")
    torch.save(net, ckpt_path)
    print('Finished Training')

    # return 100 * correct / total, running_loss_tot / 100
    return top1.avg, losses.avg


@beartype
def testval(
        ckpt_path: Path,
        net: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        visualise: Path,
):
    # net = Net()
    # net = torch.load(ckpt_path)
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    files = glob.glob(str(visualise))
    for f in files:
        os.remove(f)
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            total += labels.size(0)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # correct += (outputs.argmax(1) == labels).sum().item()
            # print(f"outputs: {outputs}")
            # print(f"labels: {labels}")
            predicted = outputs.argmax(1)
            # print(f"predicted: {predicted}")
            correct += (predicted == labels).sum().item()
            print('correct: %d' % correct)
            print('total: %d ' % total)
            print('\n')
            inputs = inputs.cpu()
            labels = labels.cpu()
            predicted = predicted.cpu()
            imshow(torchvision.utils.make_grid(inputs))
            image_path = f"data/visualise/{total/4}_{('-').join(list(map(str, labels.tolist())))}_{('-').join(list(map(str, predicted.tolist())))}.jpg"
            plt.savefig(image_path)
            os.chmod(image_path, 0o777)
    acc = 100 * correct / total
    val_loss = val_loss / 100
    return acc, val_loss
    print(f"Accuracy of the network on the validation set is: {acc} %%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default="data/spa-classify.csv?dl=0", type=Path)
    parser.add_argument('--im_dir', default="data/pictures/spa_images", type=Path)
    parser.add_argument('--ckpt_path', default="data/model.pt", type=Path)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--keep_k_most_common_labels', default=5, type=int)
    parser.add_argument('--dataset', default="five-class")
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--print_freq', default=10, type=int)
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
                                          transforms.RandomHorizontalFlip(),
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
    net = Net()
    writer = build_summary_writer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    for epoch in range(args.num_epochs):
        with BlockTimer(f"[{epoch}/{args.num_epochs} training and eval"):
            train_acc, train_loss = train(
                net=net,
                epoch=epoch,
                device=device,
                ckpt_path=args.ckpt_path,
                train_loader=train_loader,
                learning_rate=args.learning_rate,
                frequency=args.print_freq,
            )
            val_acc, val_loss = testval(
                net=net,
                ckpt_path=args.ckpt_path,
                val_loader=val_loader,
                device=device,
                visualise=args.visualise,
            )

            writer.add_scalar("train_loss", train_loss, global_step=epoch)
            writer.add_scalar("val_loss", val_loss, global_step=epoch)
            writer.add_scalar("train_acc", train_acc, global_step=epoch)
            writer.add_scalar("val_acc", val_acc, global_step=epoch)

    writer.close()
    print(paintings_train.label2idx)


if __name__ == '__main__':
    main()

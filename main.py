import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        list_of_images_folder = os.listdir(self.data_dir)
        for i, img in enumerate(csv_contents['img_file']):
            name_and_date = re.match(pattern, img)[0]
            number = re.findall(number_pattern, img)[1]
            if f"{name_and_date}-{number}.jpg" in list_of_images_folder:
                images.append(f"{self.data_dir}/{img}")
                labels.append(csv_contents['gender'][i])
            else:
                if f"{name_and_date}_{number}.jpg" in list_of_images_folder: 
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
        # img = np.array(img)
        return img, label

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(Path, transform, device):
    
    paintings_train = paintingsDataset(
            data_dir="/users/oncescu/data/pictures_project/pictures/spa_images",
        csv_file_path="/users/oncescu/data/pictures_project/spa-classify.csv?dl=0",
        split='train', transform=transform)
    imag, label = paintings_train[0]
    print(imag.size())
    train_loader = torch.utils.data.DataLoader(paintings_train,
                                             batch_size=4, shuffle=True,
                                             num_workers=4, drop_last=True)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    plt.savefig('test.jpg')
    print(' '.join('%d' % labels[j] for j in range(4)))

    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss:%.3f' %
                (epoch + 1, i + 1, running_loss/100))
                running_loss = 0.0
    
    torch.save(net, Path)
    print('Finished Training')

def testval(Path, transform, split, device):
    net = Net()
    net = torch.load(Path)
    net.eval()
    net.to(device)
    paintings = paintingsDataset(
        data_dir="/users/oncescu/data/pictures_project/pictures/spa_images",
        csv_file_path="/users/oncescu/data/pictures_project/spa-classify.csv?dl=0",
        split=split, transform=transform)

    loader = torch.utils.data.DataLoader(paintings,
                                            batch_size=4, shuffle=True,
                                            num_workers=4, drop_last=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs.float())
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()
    print('Accuracy of the network on the validation set is: %d %%' %
        (100*correct/total))
    print('correct: %d' % correct)
    print('total: %d ' % total)


def main():
    startTime = datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(256),
                            transforms.ToTensor()])
    Path = "/users/oncescu/coding/libs/pt/pictures_project/model.pt"
    train(Path, transform, device)
    testval(Path, transform, 'val', device)
    print(datetime.now() - startTime)
if __name__ == '__main__':
    main()
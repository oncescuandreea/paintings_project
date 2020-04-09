import argparse
import glob
import os
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet34
from zsvision.zs_beartype import beartype
from zsvision.zs_utils import BlockTimer

from metrics import AverageMeter, ProgressMeter, accuracy
# from network import Net
from paintings_dataset import PaintingsDataset
from utils import add_margin, imshow, build_summary_writer

@beartype
def train(
        ckpt_path: Path,
        net: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
        frequency: int,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        idx2label: dict,
        font: Path,
        optimizer: torch.optim.SGD,
):
    dataiter = iter(train_loader)
    _, images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    plt.savefig('test.jpg')
    net.train()

    net.to(device)
    criterion = nn.CrossEntropyLoss()

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
        inputs_org, inputs, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        outputs = net(inputs)
        predicted = outputs.argmax(1)

        acc1 = accuracy(output=outputs, target=labels, topk=(1,))

        loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0].item(), images.size(0))

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        inputs_org = inputs_org.cpu()
        inputs = inputs.cpu()
        labels = labels.cpu()
        predicted = predicted.cpu()
        if i % frequency == 0:
            progress.display(i)
        if i == 0:
            new_org = add_margin(img_list=inputs_org[0:8, :, :],
                                 labels=labels,
                                 predictions=predicted,
                                 margins=5,
                                 idx2label=idx2label,
                                 font=font,
                                )
            new_trans = add_margin(img_list=inputs[0:8, :, :],
                                   labels=labels,
                                   predictions=predicted,
                                   margins=5,
                                   idx2label=idx2label,
                                   font=font,
                                  )
            writer.add_image(f"Image_train_marg/%d_input_no_transform" % i,
                             torchvision.utils.make_grid(new_org),
                             epoch)
            writer.add_image(f"Image_train/%d_input_transform" % i,
                             torchvision.utils.make_grid(new_trans),
                             epoch)

    print(f" * Acc@1 {top1.avg:.3f}")
    torch.save(net, ckpt_path)
    print('Finished Training')

    return top1.avg, losses.avg


@beartype
def testval(
        ckpt_path: Path,
        net: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epoch: int,
        visualise: Path,
        frequency: int,
        writer: torch.utils.tensorboard.writer.SummaryWriter,
        idx2label: dict,
        font: Path,
):
    # net = Net()
    # net = torch.load(ckpt_path)
    net.eval()
    net.to(device)

    dataiter = iter(val_loader)
    _, images, labels = dataiter.next()

    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()

    files = glob.glob(str(visualise))
    for f in files:
        os.remove(f)
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs_org, inputs, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            # measure data loading time
            data_time.update(time.time() - end)

            outputs = net(inputs)
            predicted = outputs.argmax(1)

            acc1 = accuracy(output=outputs, target=labels, topk=(1,))

            loss = criterion(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            inputs_org = inputs_org.cpu()
            inputs = inputs.cpu()
            labels = labels.cpu()
            predicted = predicted.cpu()

            if i % frequency == 0:
                progress.display(i)
            count = 0
            # if i == 0:
            while count + 8 < len(inputs_org):
                new_org = add_margin(img_list=inputs_org[count:count+8, :, :],
                                        labels=labels,
                                        predictions=predicted,
                                        margins=5,
                                        idx2label=idx2label,
                                        font=font,
                                    )
                new_trans = add_margin(img_list=inputs[count:count+8, :, :],
                                        labels=labels,
                                        predictions=predicted,
                                        margins=5,
                                        idx2label=idx2label,
                                        font=font,
                                        )
                writer.add_image(f"Image_val_marg_{epoch}_{count/8}/%d_input_no_transform" % i,
                                torchvision.utils.make_grid(new_org),
                                epoch)
                writer.add_image(f"Image_val_{epoch}_{count/8}/%d_input_transform" % i,
                                torchvision.utils.make_grid(new_trans),
                                epoch)
                count += 8
            imshow(torchvision.utils.make_grid(inputs))
            image_path = f"data/visualise/{i}_{('-').join(list(map(str, labels.tolist())))}_{('-').join(list(map(str, predicted.tolist())))}.jpg"
            plt.savefig(image_path)
            os.chmod(image_path, 0o777)
            confusion_matrix = tf.math.confusion_matrix(np.asarray(labels),
                                                        np.asarray(predicted),
                                                        num_classes=5)
            print("Validation confusion matrix:")
            print(np.asarray(confusion_matrix))

    print(f" * Acc@1 {top1.avg:.3f}")
    return top1.avg, losses.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default="data/spa-classify.csv?dl=0", type=Path)
    parser.add_argument('--im_dir', default="data/pictures/spa_images", type=Path)
    parser.add_argument('--ckpt_path', default="data/model.pt", type=Path)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--keep_k_most_common_labels', default=5, type=int)
    parser.add_argument('--dataset', default="five-class")
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--font_type', default="/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", type=Path)
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
                                          transforms.RandomVerticalFlip(),
                                        #   transforms.Grayscale(num_output_channels=3),
                                          transforms.RandomCrop(490),
                                          transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_val = transforms.Compose([transforms.RandomCrop(490),
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    writer = build_summary_writer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model='resnet34',
    )
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
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    # net = Net()
    net = resnet34(pretrained=True)
    net._modules['fc'] = nn.Linear(512, 5, bias=True)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           patience=2,
                                                           factor=0.5)
                                                        #    cooldown=1)
    for epoch in range(args.num_epochs):
        with BlockTimer(f"[{epoch}/{args.num_epochs} training and eval"):
            train_acc, train_loss = train(
                net=net,
                epoch=epoch,
                device=device,
                ckpt_path=args.ckpt_path,
                train_loader=train_loader,
                frequency=args.print_freq,
                writer=writer,
                idx2label=dict(map(reversed, paintings_train.label2idx.items())),
                font=args.font_type,
                optimizer=optimizer,
            )
            val_acc, val_loss = testval(
                net=net,
                ckpt_path=args.ckpt_path,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                visualise=args.visualise,
                frequency=args.print_freq,
                writer=writer,
                idx2label=dict(map(reversed, paintings_train.label2idx.items())),
                font=args.font_type,
            )
            scheduler.step(val_loss)

            writer.add_scalar("train_loss", train_loss, global_step=epoch)
            writer.add_scalar("val_loss", val_loss, global_step=epoch)
            writer.add_scalar("train_acc", train_acc, global_step=epoch)
            writer.add_scalar("val_acc", val_acc, global_step=epoch)

    writer.close()
    print(paintings_train.label2idx)


if __name__ == '__main__':
    main()

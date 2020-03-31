from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from zsvision.zs_beartype import beartype


def stratify(labels):
    dict_of_lists = defaultdict(list)
    for i, el in enumerate(labels):
        dict_of_lists[el].append(i)
    return dict_of_lists

@beartype
def add_margin(
        img_list: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        margins: int,
        idx2label: dict,
        font: Path,
):
    transformToPil = transforms.Compose([transforms.ToPILImage()])
    transformToTensor = transforms.Compose([transforms.ToTensor()])
    new_images = []
    for k, img in enumerate(img_list):
        if labels[k] == predictions[k]:
            color = "green"
        else:
            color = "red"
        pil_img = transformToPil(img)
        width, height = pil_img.size
        bottom = top = right = left = margins
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))

        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 15)
        add_text = ImageDraw.Draw(result)
        rgb_mean = list(map(int, [np.mean(np.asarray(result)[:, :, 0]),
                                  np.mean(np.asarray(result)[:, :, 1]),
                                  np.mean(np.asarray(result)[:, :, 2])]))
        rgb_mean = [(el + 150) % 255 for el in rgb_mean]
        add_text.text((10, 10), text=f"label:{idx2label[labels[k].item()]}",
                       font=font, fill=(rgb_mean[0], rgb_mean[1], rgb_mean[2]), align="center")
        add_text.text((10, 30), text=f"prediction:{idx2label[predictions[k].item()]}",
                      font=font, fill=(rgb_mean[0], rgb_mean[1], rgb_mean[2]), align="center")
        result = transformToTensor(result)
        new_images.append(result)
    return new_images

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
def build_summary_writer(
        learning_rate: float,
        batch_size: int,
        model: str,
) -> SummaryWriter:
    timestamp = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    log_dir = Path("data/logs") / f"model-{model}-lr-{learning_rate}-bs-{batch_size}" / timestamp
    log_dir.mkdir(exist_ok=True, parents=True)
    return SummaryWriter(str(log_dir))

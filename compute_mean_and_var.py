"""Compute dataset statistics for the paintings

Statistics for the 124-painting subset covering five media:
    RGB mean: [0.5827, 0.5678, 0.5294]
    RGB std: [0.2707, 0.2595, 0.2776]
"""
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from zsvision.zs_utils import BlockTimer

from main import PaintingsDataset


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
        ims, _ = next(iter(loader))
        ims = ims.refine_names("N", "C", "H", "W")
        rgb_mean = ims.mean(("N", "H", "W"))
        rgb_std = ims.std(("N", "H", "W"))

    print(f"Computed RGB (mean): {rgb_mean}")
    print(f"Computed RGB (std): {rgb_std}")


if __name__ == '__main__':
    main()

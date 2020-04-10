"""Compute dataset statistics for the paintings

Statistics for the 124-painting subset covering five media:
    RGB mean: [0.5827, 0.5678, 0.5294]
    RGB std: [0.2707, 0.2595, 0.2776]

Statistics for the 20668-painting subset artuk covering five media:
    RGB mean: [0.4971, 0.4365, 0.3489]
    RGB std: [0.2594, 0.2457, 0.2373]
"""
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from zsvision.zs_utils import BlockTimer

from paintings_dataset import PaintingsDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',
                        default="/scratch/shared/beegfs/oncescu/pictures_project/artuk/artuk_lists",
                        type=Path)
    parser.add_argument('--im_dir',
                        default="/scratch/shared/beegfs/oncescu/pictures_project/artuk/paintings",
                        type=Path)
    parser.add_argument('--ckpt_path', default="data/modelartuk.pt", type=Path)
    parser.add_argument('--im_suffix', default=".jpg",
                        help="the suffix for images in the dataset")
    args = parser.parse_args()

    with BlockTimer("computing statistics for dataset"):
        transform = transforms.Compose([transforms.CenterCrop(69),
                                        transforms.Resize(224),
                                        transforms.ToTensor()])
        dataset = PaintingsDataset(
            data_dir=args.im_dir,
            split="train",
            transform=transform,
            csv_path=args.csv_path,
            im_suffix=args.im_suffix,
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        _, ims, _ = next(iter(loader))
        ims = ims.refine_names("N", "C", "H", "W")
        rgb_mean = ims.mean(("N", "H", "W"))
        rgb_std = ims.std(("N", "H", "W"))

    print(f"Computed RGB (mean): {rgb_mean}")
    print(f"Computed RGB (std): {rgb_std}")


if __name__ == '__main__':
    main()

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def get_min_size(csv_contents_path, data_dir):
    with open(csv_contents_path, "r") as f:
        csv_contents = pd.read_csv(f)
    for i, el in enumerate(csv_contents.iloc[:, 1]):
            img = Image.open(data_dir / f"{el}.jpg")
            min_size = np.min(img.size)
            max_size = np.max(img.size)
            if i == 0:
                min_size_general = min_size
                max_size_general = max_size
            else:
                if min_size < min_size_general:
                    min_size_general = min_size
                if max_size > max_size_general:
                    max_size_general = max_size
    return min_size_general, max_size_general

def main():
    data_dir = Path("/scratch/shared/beegfs/oncescu/pictures_project/artuk/paintings")
    csv_contents_path = Path("/scratch/shared/beegfs/oncescu/pictures_project/artuk/artuk_lists/artuk_train.csv")
    min_size_general, max_size_general = get_min_size(csv_contents_path, data_dir)
    print(f"Minimum pixel size of a painting is {min_size_general}")
    print(f"Maximum pixel size of a painting is {max_size_general}")


if __name__ == '__main__':
    main()

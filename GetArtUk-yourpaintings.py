import csv
import re

from collections import Counter
from pathlib import Path

def get_painters_count(location, painters):
    w = csv.writer(open(location / "artuk_painters_count.csv", "w"))
    for key, val in Counter(painters).most_common():
        w.writerow([key, val])

def get_painter_names(location, painters):
    painters_text = ('\n').join(painters)
    with open(location / "painters_artuk.txt", 'a') as f:
        f.write(painters_text)

def get_csv_info(location):
    painters = []
    with open(location / "yp_alts.txt", "r") as f:
        for row in f:
            info_components = row.split(',')
            last_component = re.sub("[\s\n]", "", info_components[-1])
            
            if len(info_components) == 2:
                painter_name = re.sub("[\n]", "", info_components[-1])[1:]
            elif last_component >= '1000' and last_component <= '9999':
                painter_name = info_components[-2][1:]
            else:
                painter_name = re.sub("[\n]", "", info_components[-1])[1:]
            painters.append(painter_name)
    return painters

def main():
    location = Path("/scratch/shared/beegfs/oncescu/pictures_project/artuk/artuk_lists")
    painters = get_csv_info(location)


if __name__ == '__main__':
    main()

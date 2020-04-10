import argparse
import csv
from pathlib import Path

def train_test_val(location, train_split, val_split):
    '''
    Function creates 3 .csv files containing the painter name, index corresponding to
    picture number and the link of that picture
    Inputs:
        location - location of the folder containing the existend metadata
        train_split - percentage of the data used for training
        val_split - percentage of data used for validaiton
    '''
    list_of_skipped_entries = ["", " unknown artist", " British (English) School",
                            " British School", " French School", " Italian School",
                            " Dutch School", " Flemish School", " British (Scottish) School",
                            " Italian (Venetian) School", " Spanish School",
                            " Netherlandish School", " British (Welsh) School",
                            " German School"]

    csv_train = csv.writer(open(location / "artuk_train1.csv", "w"))
    csv_val = csv.writer(open(location / "artuk_val1.csv", "w"))
    csv_test = csv.writer(open(location/ "artuk_test1.csv", "w"))

    with open(location / "artuk_painters_count.csv") as f:
        for row_f in f:
            artist, no_paintings = row_f.split(",")
            no_paintings = int(no_paintings.rstrip())
            if artist not in list_of_skipped_entries:
                if no_paintings >= 100:
                    found_group = False
                    count = 0
                    train_count = int(no_paintings * train_split)
                    val_count = int(no_paintings * (val_split + train_split))
                    with open(location / "artuk_painters_links.csv", "r") as g:
                        for row_g in g:
                            _, name, link, position = row_g.split(",")
                            position = position.rstrip()
                            if name == artist:
                                if count <= train_count:
                                    csv_train.writerow([artist, position, link])
                                elif count > train_count and count <= val_count:
                                    csv_val.writerow([artist, position, link])
                                else:
                                    csv_test.writerow([artist, position, link])
                                count += 1
                                if found_group is False:
                                    found_group = True
                            else:
                                if name != artist and found_group is True:
                                    # reached end of the group of paintings from
                                    # the same painter
                                    break
                else:
                    break
    g.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lists_locations',
                        default="/scratch/shared/beegfs/oncescu/pictures_project/artuk/artuk_lists",
                        type=Path)
    parser.add_argument('--train_split', default=0.65, type=float)
    parser.add_argument('--val_split', default=0.2, type = float)
    args = parser.parse_args()

    train_test_val(args.lists_locations, args.train_split, args.val_split)

if __name__ == '__main__':
    main()

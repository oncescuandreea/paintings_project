import argparse
import csv
from pathlib import Path

def train_test_val(location, train_split, val_split, min_number_paintings, equal='False'):
    '''
    Function creates 3 .csv files containing the painter name, index corresponding to
    picture number and the link of that picture
    Inputs:
        location - location of the folder containing the existend metadata
        train_split - percentage of the data used for training
        val_split - percentage of data used for validation
    '''
    list_of_skipped_entries = ["", "unknown artist", "British (English) School",
                            "British School", "French School", "Italian School",
                            "Dutch School", "Flemish School", "British (Scottish) School",
                            "Italian (Venetian) School", "Spanish School",
                            "Netherlandish School", "British (Welsh) School",
                            "German School", "Chinese School", "Italian (Florentine) School",
                            "Northern Italian School", "Victoria Hospital Staff and Patients",
                            "Irish School", "Italian (Neapolitan) School",
                            "Italian (Lombard) School", "Italian (Roman) School",
                            "Italian (Bolognese) School", "Anglo/Dutch School"]

    csv_train = csv.writer(open(location / "artuk_train.csv", "w"))
    csv_val = csv.writer(open(location / "artuk_val.csv", "w"))
    csv_test = csv.writer(open(location/ "artuk_test.csv", "w"))

    with open(location / "artuk_painters_count.csv") as f:
        for row_f in f:
            artist, no_paintings = row_f.split(",")
            no_paintings = int(no_paintings.rstrip())
            if artist not in list_of_skipped_entries:
                if no_paintings >= min_number_paintings:
                    found_group = False
                    count = 0
                    if equal is False:
                        train_count = int(no_paintings * train_split)
                        val_count = int(no_paintings * (val_split + train_split))
                        test_count = no_paintings
                    else:
                        train_count = int(min_number_paintings * train_split)
                        val_count = int(min_number_paintings * (val_split + train_split))
                        test_count = min_number_paintings
                    with open(location / "artuk_painters_links.csv", "r") as g:
                        for row_g in g:
                            _, name, link, position = row_g.split(",")
                            position = position.rstrip()
                            if name == artist:
                                if count <= train_count:
                                    csv_train.writerow([artist, position, link])
                                elif train_count < count <= val_count:
                                    csv_val.writerow([artist, position, link])
                                elif val_count < count < test_count:
                                    csv_test.writerow([artist, position, link])
                                else:
                                    break
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
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--min_number_paintings', default=30, type=int,
                        help='select only painters with more than this number of paintings')
    parser.add_argument('--equal', default='True', type=bool,
                        help='number of examples per painter are equal')
    args = parser.parse_args()

    train_test_val(args.lists_locations, args.train_split,
                   args.val_split, args.min_number_paintings,
                   args.equal)

if __name__ == '__main__':
    main()

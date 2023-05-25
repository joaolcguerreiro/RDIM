import shutil
import os
import csv
from glob import glob


def main():
    # Open CSV metadata file
    try:
        with open(max(glob(os.path.join('..', '..', 'data', 'manifest*', 'metadata.csv')), key=os.path.getmtime), newline='') as csvfile:
            reader = csv.reader(csvfile)

            # Skip first row (Column names)
            next(reader)

            # Desired index
            modality_index = 10
            path_index = 15

            # Parse path function
            parse_path = lambda path: max(glob(os.path.join('..', '..', 'data', 'manifest*', os.path.dirname(path)[2:])), key=os.path.getmtime)

            # Clean data
            for row in reader:
                if row[modality_index] != 'CT':
                    shutil.rmtree(parse_path(row[path_index]))
    except Exception as e:
        print("Download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    main()

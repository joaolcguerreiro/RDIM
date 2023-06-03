import shutil
import argparse
import os
import csv
from glob import glob


def main():
    try:
        # Open CSV metadata file
        with open(os.path.join(args.dir, 'metadata.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)

            # Skip first row (Column names)
            next(reader)

            # Desired index
            modality_index = 10
            path_index = 15

            # Parse path function
            parse_path = lambda path: os.path.join(args.dir, os.path.dirname(path)[2:])

            # Clean data
            for row in reader:
                if row[modality_index] != 'CT':
                    shutil.rmtree(parse_path(row[path_index]))
    except Exception as e:
        print("Download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', default=max((path for path in glob(os.path.join('..', '..', 'data', 'manifest*')) if os.path.isdir(path)), key=os.path.getmtime, default=None), help='Path to directory with data', type=str)
    args = parser.parse_args()

    main()

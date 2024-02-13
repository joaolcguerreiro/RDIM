import shutil
import argparse
import os
import csv


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
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

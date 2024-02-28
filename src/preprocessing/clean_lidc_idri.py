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
            file_path_index = 15

            # Clean data
            for row in reader:
                if row[modality_index] != 'CT':
                    shutil.rmtree(os.path.join(args.dir, os.path.dirname(row[file_path_index][2:])))

        # Remove metadata and license files
        license_file_path = os.path.join(args.dir, 'LIDC-IDRI', 'LICENSE')
        metadata_file_path = os.path.join(args.dir, 'metadata.csv')

        if os.path.exists(license_file_path):
            os.remove(license_file_path)

        if os.path.exists(metadata_file_path):
            os.remove(metadata_file_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

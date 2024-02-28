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
            type_index = 11
            file_path_index = 15

            # Clean data
            for row in reader:
                if row[modality_index] != 'CT':
                    shutil.rmtree(os.path.join(args.dir, os.path.dirname(row[file_path_index][2:])))
                elif row[type_index] != 'CT Image Storage':
                    dir_path = os.path.join(args.dir, os.path.dirname(row[file_path_index][2:]))

                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)

        # Fetch all data directory paths
        parent_dir_paths = sorted(glob(os.path.join(args.dir, 'LDCT-and-Projection-data', '*')))

        # Discard cases were any counterpart is missing
        for parent_dir_path in parent_dir_paths:
            # Pass non-directory files
            if not os.path.isdir(parent_dir_path):
                continue

            # Get path of the child directory
            child_dir_path = os.path.join(parent_dir_path, [child_file_path for child_file_path in os.listdir(parent_dir_path) if child_file_path != '.DS_Store' and os.path.isdir(os.path.join(parent_dir_path, child_file_path))][0])

            if len([file_path for file_path in os.listdir(child_dir_path) if file_path != '.DS_Store']) == 2:
                shutil.move(child_dir_path, args.dir)
                os.rename(os.path.join(args.dir, os.path.basename(child_dir_path)), os.path.join(args.dir, os.path.basename(parent_dir_path)))

            # Remove redundant parent directory
            shutil.rmtree(parent_dir_path)

        # Remove metadata file and old root directory
        metadata_file_path = os.path.join(args.dir, 'metadata.csv')
        redundant_dir_path = os.path.join(args.dir, 'LDCT-and-Projection-data')

        if os.path.exists(metadata_file_path):
            os.remove(metadata_file_path)

        if os.path.exists(redundant_dir_path):
            shutil.rmtree(redundant_dir_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

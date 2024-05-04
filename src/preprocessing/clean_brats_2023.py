import shutil
import argparse
import os
from glob import glob


def main():
    try:
        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*')))

        # Discard non T1-weighted without contrast (t1n) scans:
        for parent_dir_path in dataset_file_paths:
            # Pass non-directory files
            if not os.path.isdir(parent_dir_path):
                continue

            # Get path of the child directory
            t1n_file_path = sorted(glob(os.path.join(parent_dir_path, '*t1n.nii.gz')))[0]

            shutil.move(t1n_file_path, args.dir)

            # Remove redundant parent directory
            shutil.rmtree(parent_dir_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

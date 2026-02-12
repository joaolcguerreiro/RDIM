import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Unzip Rain100H dataset zip file
        dataset_zip_file_path = os.path.join(args.dir, 'archive.zip')

        if os.path.exists(dataset_zip_file_path):
            with zipfile.ZipFile(dataset_zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove zip file
            os.remove(dataset_zip_file_path)

        # Rename directories
        for old_split_name, new_split_name in [('train', 'training'), ('test_b', 'testing')]:
            old_split_path = os.path.join(args.dir, old_split_name)
            new_split_path = os.path.join(args.dir, new_split_name)

            if os.path.exists(old_split_path) and not os.path.exists(new_split_path):
                os.rename(old_split_path, new_split_path)

            if os.path.exists(new_split_path):
                for old_pair_name, new_pair_name in [('gt', 'HR'), ('data', 'LR')]:
                    old_pair_path = os.path.join(new_split_path, old_pair_name)
                    new_pair_path = os.path.join(new_split_path, new_pair_name)

                    if os.path.exists(old_pair_path) and not os.path.exists(new_pair_path):
                        os.rename(old_pair_path, new_pair_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

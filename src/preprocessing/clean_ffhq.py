import os
import shutil
import argparse
from glob import glob


def main():
    try:
        # Remove license file
        license_file_path = os.path.join(args.dir, 'LICENSE.txt')

        if os.path.exists(license_file_path):
            os.remove(license_file_path)

        # Move the data from the redundant intermediary directories to the root directory
        src_dir_paths = [sub_dir for sub_dir in sorted(glob(os.path.join(args.dir, '*'))) if os.path.isdir(sub_dir)]
        dest_dir_path = args.dir

        for src_dir_path in src_dir_paths:
            for file_path in os.listdir(src_dir_path):
                if file_path != ".DS_Store":
                    shutil.move(os.path.join(src_dir_path, file_path), dest_dir_path)

            # Remove redundant intermediary directory
            shutil.rmtree(src_dir_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

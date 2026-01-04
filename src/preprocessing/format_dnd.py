import os
import shutil
import argparse
from glob import glob


def main():
    try:
        # Define testing directory
        test_dir_path = os.path.join(args.dir, 'testing')

        # Define LR directory
        lr_dir_path = os.path.join(test_dir_path, 'LR')

        # Remove and create LR directory
        if os.path.exists(lr_dir_path):
            shutil.rmtree(lr_dir_path)

        os.makedirs(lr_dir_path)

        # Fetch all test files
        file_paths = sorted(file_path for file_path in glob(os.path.join(test_dir_path, '*')) if os.path.basename(file_path) != 'LR')

        # Parse files
        for file_path in file_paths:
            if file_path.endswith('.mat'):
                shutil.move(file_path, lr_dir_path)
            else:
                os.remove(file_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

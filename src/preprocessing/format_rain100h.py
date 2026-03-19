import os
import shutil
import argparse
from glob import glob


def main():
    try:
        # Define training and testing directories
        train_dir_path = os.path.join(args.dir, 'training')
        test_dir_path = os.path.join(args.dir, 'testing')

        # Restructure dataset
        for dir_path in [train_dir_path, test_dir_path]:
            # Remove and create LR and HR directories
            lr_dir_path = os.path.join(dir_path, 'LR')
            hr_dir_path = os.path.join(dir_path, 'HR')

            if os.path.exists(lr_dir_path):
                shutil.rmtree(lr_dir_path)

            os.makedirs(lr_dir_path)

            if os.path.exists(hr_dir_path):
                shutil.rmtree(hr_dir_path)

            os.makedirs(hr_dir_path)

            # Fetch all data file paths
            dataset_file_paths = sorted(file_path for file_path in glob(os.path.join(dir_path, '*')) if os.path.basename(file_path) not in ['.DS_Store', 'LR', 'HR'])

            for file_path in dataset_file_paths:
                filename = os.path.basename(file_path)

                if filename.startswith('norain-'):
                    shutil.move(file_path, os.path.join(dir_path, 'HR'))
                elif filename.startswith('rain-'):
                    shutil.move(file_path, os.path.join(dir_path, 'LR'))
                else:
                    raise NotImplementedError('Invalid filename')

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

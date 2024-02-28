import shutil
import os
import argparse
from glob import glob
from sklearn.model_selection import train_test_split


def main():
    try:
        # Remove and create training, validation, and testing directories
        train_dir_path = os.path.join(args.dir, 'training')
        val_dir_path = os.path.join(args.dir, 'validation')
        test_dir_path = os.path.join(args.dir, 'testing')

        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

            os.makedirs(dir_path)

        # Fetch all data file paths
        dataset_file_paths = sorted(file_path for file_path in glob(os.path.join(args.dir, '*')) if os.path.basename(file_path) not in ['.DS_Store', 'training', 'validation', 'testing'])

        # Split dataset
        train, test = train_test_split(dataset_file_paths, test_size=(args.test_size * 2 if args.val else args.test_size), shuffle=True, random_state=20)
        val, test = train_test_split(test, test_size=0.5, shuffle=True, random_state=20) if args.val else ([], test)

        for split, dir_path in zip([train, val, test], [train_dir_path, val_dir_path, test_dir_path]):
            for file_path in split:
                shutil.move(file_path, dir_path)
    except Exception as e:
        print("Check directory paths. If error remains download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Split data')
    parser.add_argument('--dir', required=True, help='Path to directory with data', type=str)
    parser.add_argument('--val', default=True, help='Build validation split', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--test-size', default=0.05, help='Fraction of data for evaluation', type=float)
    args = parser.parse_args()

    main()

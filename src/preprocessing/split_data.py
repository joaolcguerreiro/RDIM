import shutil
import os
import argparse
from glob import glob
from sklearn.model_selection import train_test_split


def clean_directory(dir_path):
    for file in sorted(glob(os.path.join(dir_path, '*'))):
        os.remove(file)


def main():
    # Path to training, validation, and testing directories
    training_dir = args.train
    validation_dir = args.val
    testing_dir = args.test

    # Clean training and testing directories
    clean_directory(training_dir)
    clean_directory(validation_dir)
    clean_directory(testing_dir)

    # Fetch all training data file paths
    dataset_files = sorted(glob(os.path.join(args.dir, '*')))

    # Split dataset
    train, val = train_test_split(dataset_files, test_size=(0.20 if args.val else 0.10), shuffle=True, random_state=20)
    val, test = train_test_split(val, test_size=0.5, shuffle=True, random_state=20) if args.val else ([], val)

    for (split, directory) in [(train, training_dir), (val, validation_dir), (test, testing_dir)]:
        for file in split:
            _, file_name = os.path.split(file)
            shutil.copy(file, os.path.join(directory, file_name))


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Split training data')
    parser.add_argument('--val', default=True, help='Build validation split', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--dir', default=os.path.join('..', '..', 'data', 'LIDC-IDRI'), help='Path to directory with training data', type=str)
    parser.add_argument('--train', default=os.path.join('..', '..', 'data', 'training'), help='Path to destination training directory', type=str)
    parser.add_argument('--val', default=os.path.join('..', '..', 'data', 'validation'), help='Path to destination validation directory', type=str)
    parser.add_argument('--test', default=os.path.join('..', '..', 'data', 'testing'), help='Path to destination testing directory', type=str)
    args = parser.parse_args()

    main()

import shutil
import os
import argparse
from glob import glob
from sklearn.model_selection import train_test_split


def main():
    try:
        # Remove training, validation, and testing directories
        shutil.rmtree(args.train_dir) if os.path.exists(args.train_dir) else None
        shutil.rmtree(args.val_dir) if os.path.exists(args.val_dir) else None
        shutil.rmtree(args.test_dir) if os.path.exists(args.test_dir) else None

        # Create training, validation, and testing directories
        os.makedirs(args.train_dir, exist_ok=True)
        os.makedirs(args.val_dir, exist_ok=True) if args.val else None
        os.makedirs(args.test_dir, exist_ok=True)

        # Fetch all training data file paths
        dataset_files = sorted(glob(os.path.join(args.dir, '*')))

        # Split dataset
        train, val = train_test_split(dataset_files, test_size=(0.10 if args.val else 0.05), shuffle=True, random_state=20)
        val, test = train_test_split(val, test_size=0.5, shuffle=True, random_state=20) if args.val else ([], val)

        for (split, directory) in [(train, args.train_dir), (val, args.val_dir), (test, args.test_dir)]:
            for file in split:
                _, file_name = os.path.split(file)
                shutil.copy(file, os.path.join(directory, file_name))
    except Exception as e:
        print("Check directory paths. If error remains download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Split data')
    parser.add_argument('--val', default=True, help='Build validation split', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--dir', default=os.path.join('..', '..', 'data', 'LIDC-IDRI'), help='Path to directory with data', type=str)
    parser.add_argument('--train-dir', default=os.path.join('..', '..', 'data', 'training'), help='Path to destination training directory', type=str)
    parser.add_argument('--val-dir', default=os.path.join('..', '..', 'data', 'validation'), help='Path to destination validation directory', type=str)
    parser.add_argument('--test-dir', default=os.path.join('..', '..', 'data', 'testing'), help='Path to destination testing directory', type=str)
    args = parser.parse_args()

    main()

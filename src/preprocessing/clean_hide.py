import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Unzip HIDE dataset zip file
        dataset_zip_file_path = os.path.join(args.dir, 'HIDE_dataset.zip')

        if os.path.exists(dataset_zip_file_path):
            with zipfile.ZipFile(dataset_zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove zip file
            os.remove(dataset_zip_file_path)

        # Extract all directories
        parent_dir_path = os.path.join(args.dir, 'HIDE_dataset')
        dataset_dir_paths = sorted(item_path for item_path in glob(os.path.join(parent_dir_path, '*')) if os.path.isdir(item_path))

        for dir_path in dataset_dir_paths:
            shutil.move(dir_path, args.dir)

        if os.path.exists(parent_dir_path):
            shutil.rmtree(parent_dir_path)

        # Fetch all testing item paths
        testing_item_paths = sorted(glob(os.path.join(args.dir, 'test', '*')))

        # Remove redundant intermediary testing directories
        for testing_item_path in testing_item_paths:
            if os.path.isfile(testing_item_path) and os.path.basename(testing_item_path).endswith('.png'):
                continue
            elif not os.path.isdir(testing_item_path):
                os.remove(testing_item_path)
            else:
                for file_path in glob(os.path.join(testing_item_path, '*.png')):
                    shutil.move(file_path, os.path.join(args.dir, 'test'))

                # Remove redundant intermediary directory
                shutil.rmtree(testing_item_path)

        # Rename directories
        for old_name, new_name in [('train', 'training'), ('test', 'testing')]:
            old_path = os.path.join(args.dir, old_name)
            new_path = os.path.join(args.dir, new_name)

            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

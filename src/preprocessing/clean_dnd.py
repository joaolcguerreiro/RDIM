import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Define testing directory
        test_dir_path = os.path.join(args.dir, 'testing')
        os.makedirs(test_dir_path, exist_ok=True)

        # Unzip DND dataset zip file
        dataset_zip_file_path = os.path.join(args.dir, 'dnd_2017.zip')

        if os.path.exists(dataset_zip_file_path):
            with zipfile.ZipFile(dataset_zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove zip file
            os.remove(dataset_zip_file_path)

        # Move the data from the redundant intermediary directories to the root directory
        src_dir_path = next(iter(glob(os.path.join(args.dir, '**', 'images_srgb'), recursive=True)), None)
        dest_dir_path = test_dir_path

        if src_dir_path and os.path.exists(src_dir_path):
            for file_path in os.listdir(src_dir_path):
                if file_path != ".DS_Store":
                    shutil.move(os.path.join(src_dir_path, file_path), dest_dir_path)

        # Delete redundant intermediary directories or files if present
        items = os.listdir(args.dir)

        for item in items:
            if item not in ['testing']:
                item_path = os.path.join(args.dir, item)

                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Unzip Dunhuang Grottoes Painting dataset zip file
        dataset_zip_file_path = os.path.join(args.dir, 'archive.zip')

        if os.path.exists(dataset_zip_file_path):
            with zipfile.ZipFile(dataset_zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove zip file
            os.remove(dataset_zip_file_path)

        # Move the data from the redundant intermediary directory to the root directory
        src_dir_path = next(iter(glob(os.path.join(args.dir, '*'))), None)
        dest_dir_path = args.dir

        if src_dir_path and os.path.exists(src_dir_path):
            for dir_path in os.listdir(src_dir_path):
                if dir_path != ".DS_Store":
                    shutil.move(os.path.join(src_dir_path, dir_path), dest_dir_path)

            # Remove redundant intermediary directory
            shutil.rmtree(src_dir_path)

        # Rename directories
        for old_name, new_name in [('train', 'training'), ('test', 'testing')]:
            old_path = os.path.join(args.dir, old_name)
            new_path = os.path.join(args.dir, new_name)

            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)

            # Remove mask directories
            mask_dir_path = os.path.join(new_path, f'{old_name}_mask')

            if os.path.exists(mask_dir_path):
                shutil.rmtree(os.path.join(new_path, f'{old_name}_mask'))

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

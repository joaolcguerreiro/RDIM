import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Define training and testing directories
        train_dir_path = os.path.join(args.dir, 'training')
        test_dir_path = os.path.join(args.dir, 'testing')

        for dir_path in [train_dir_path, test_dir_path]:
            os.makedirs(dir_path, exist_ok=True)

        # Unzip combined dataset zip file
        combined_dataset_zip_file_path = os.path.join(args.dir, 'combined.zip')

        if os.path.exists(combined_dataset_zip_file_path):
            with zipfile.ZipFile(combined_dataset_zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove file
            os.remove(combined_dataset_zip_file_path)

        # Unzip non-combined dataset zip file
        dataset_zip_file_paths = glob(os.path.join(args.dir, "SIDD_*.zip"))

        for zip_file_path in dataset_zip_file_paths:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove file
            os.remove(zip_file_path)

        # Move the data from the redundant intermediary directories to the root directory
        src_dir_path = next(iter(glob(os.path.join(args.dir, '**', 'SIDD_*', 'Data'), recursive=True)), None)
        dest_dir_path = train_dir_path

        if src_dir_path and os.path.exists(src_dir_path):
            for file_path in os.listdir(src_dir_path):
                if file_path != ".DS_Store":
                    shutil.move(os.path.join(src_dir_path, file_path), dest_dir_path)

            # Remove redundant intermediary directory
            shutil.rmtree(os.path.dirname(src_dir_path))

        # Move validation MAT files
        src_dir_path = args.dir
        dest_dir_path = test_dir_path

        for mat_file in ["ValidationGtBlocksSrgb.mat", "ValidationNoisyBlocksSrgb.mat"]:
            src_file_path = os.path.join(src_dir_path, mat_file)

            if os.path.exists(src_file_path):
                shutil.move(src_file_path, dest_dir_path)

        # Delete redundant intermediary directories or files if present
        items = os.listdir(args.dir)

        for item in items:
            if item not in ['training', 'testing']:
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

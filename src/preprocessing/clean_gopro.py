import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Unzip GoPro dataset zip file
        dataset_zip_file_path = os.path.join(args.dir, 'GOPRO_Large.zip')

        if os.path.exists(dataset_zip_file_path):
            with zipfile.ZipFile(dataset_zip_file_path, 'r') as zip_file:
                zip_file.extractall(path=args.dir)

            # Remove zip file
            os.remove(dataset_zip_file_path)

        # Rename directories
        for old_name, new_name in [('train', 'training'), ('test', 'testing')]:
            old_path = os.path.join(args.dir, old_name)
            new_path = os.path.join(args.dir, new_name)

            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)

        # Remove all blur gamma folders
        for split in ['training', 'testing']:
            # Fetch all directory paths
            scene_dir_paths = sorted(glob(os.path.join(args.dir, split, '*')))

            for scene_dir_path in scene_dir_paths:
                if not os.path.isdir(scene_dir_path):
                    os.remove(scene_dir_path)
                    continue

                blur_gamma_dir_path = os.path.join(scene_dir_path, 'blur_gamma')

                if os.path.exists(blur_gamma_dir_path):
                    shutil.rmtree(blur_gamma_dir_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

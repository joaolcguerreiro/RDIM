import os
import shutil
import argparse
from glob import glob


def main():
    try:
        # Define training, validation and testing directories
        train_dir_path = os.path.join(args.dir, 'training')
        val_dir_path = os.path.join(args.dir, 'validation')
        test_dir_path = os.path.join(args.dir, 'testing')

        # Restructure dataset
        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            if not os.path.exists(dir_path):
                continue

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

            # Move files to LR and HR directories
            for image_dir_path in dataset_file_paths:
                for file_path in sorted(glob(os.path.join(image_dir_path, '*'))):
                    if "GT" in file_path:
                        # Define old image paths
                        lr_img_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace("GT", "NOISY"))
                        hr_img_path = file_path

                        # Define new image paths
                        new_lr_img_file_path = os.path.join(lr_dir_path, os.path.basename(file_path).replace("GT_", ""))
                        new_hr_img_file_path = os.path.join(hr_dir_path, os.path.basename(file_path).replace("GT_", ""))

                        shutil.copy(lr_img_path, new_lr_img_file_path)
                        shutil.copy(hr_img_path, new_hr_img_file_path)

        # Delete old directories
        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            # Fetch all data file paths
            dataset_file_paths = sorted(file_path for file_path in glob(os.path.join(dir_path, '*')) if os.path.basename(file_path) not in ['.DS_Store', 'LR', 'HR'])

            for volume_dir_path in dataset_file_paths:
                # Remove old directory
                shutil.rmtree(volume_dir_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

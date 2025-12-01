import os
import shutil
import argparse
import numpy as np
import scipy.io as sio
from glob import glob
from PIL import Image


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

        # Parse MAT files
        lr_mat_file_path = os.path.join(test_dir_path, "ValidationNoisyBlocksSrgb.mat")
        hr_mat_file_path = os.path.join(test_dir_path, "ValidationGtBlocksSrgb.mat")

        if os.path.exists(lr_mat_file_path) and os.path.exists(hr_mat_file_path):
            lr_data = sio.loadmat(lr_mat_file_path)["ValidationNoisyBlocksSrgb"]
            hr_data = sio.loadmat(hr_mat_file_path)["ValidationGtBlocksSrgb"]

            num_imgs = lr_data.shape[0]
            num_blocks = lr_data.shape[1]

            for img_index in range(num_imgs):
                for block_index in range(num_blocks):
                    lr_img = lr_data[img_index][block_index].astype(np.uint8)
                    hr_img = hr_data[img_index][block_index].astype(np.uint8)

                    Image.fromarray(lr_img).save(os.path.join(test_dir_path, 'LR', f"{img_index * num_blocks + block_index:05d}.png"))
                    Image.fromarray(hr_img).save(os.path.join(test_dir_path, 'HR', f"{img_index * num_blocks + block_index:05d}.png"))

            # Remove .mat files after extraction
            os.remove(lr_mat_file_path)
            os.remove(hr_mat_file_path)

        # Fetch all train data file paths
        train_dataset_file_paths = sorted(file_path for file_path in glob(os.path.join(train_dir_path, '*')) if os.path.basename(file_path) not in ['.DS_Store', 'LR', 'HR'])

        # Move files to LR and HR directories
        for image_dir_path in train_dataset_file_paths:
            for file_path in sorted(glob(os.path.join(image_dir_path, '*'))):
                if "GT" in file_path:
                    # Define old image paths
                    lr_img_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace("GT", "NOISY"))
                    hr_img_path = file_path

                    filename = os.path.basename(file_path)
                    prefix = os.path.basename(image_dir_path)[:4] + '_' if not filename[:4].isdigit() else ''

                    # Define new image paths
                    new_lr_img_file_path = os.path.join(train_dir_path, 'LR', prefix + filename.replace("GT_", ""))
                    new_hr_img_file_path = os.path.join(train_dir_path, 'HR', prefix + filename.replace("GT_", ""))

                    shutil.copy(lr_img_path, new_lr_img_file_path)
                    shutil.copy(hr_img_path, new_hr_img_file_path)

        # Delete old directories in training directory
        for volume_dir_path in train_dataset_file_paths:
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

import os
import shutil
import argparse
from glob import glob


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

            # Fetch all scene directories
            scene_dir_paths = sorted(scene_dir_path for scene_dir_path in glob(os.path.join(dir_path, '*')) if os.path.isdir(scene_dir_path) and os.path.basename(scene_dir_path) not in ['LR', 'HR'])

            # Move files to LR and HR directories
            for scene_dir_path in scene_dir_paths:
                scene_name = os.path.basename(scene_dir_path)

                lr_src_dir_path = os.path.join(scene_dir_path, 'blur')
                hr_src_dir_path = os.path.join(scene_dir_path, 'sharp')

                if os.path.exists(lr_src_dir_path) and os.path.exists(hr_src_dir_path):
                    for lr_img_path, hr_img_path in zip(sorted(glob(os.path.join(lr_src_dir_path, '*.png'))), sorted(glob(os.path.join(hr_src_dir_path, '*.png')))):
                        shutil.move(lr_img_path, os.path.join(lr_dir_path, f"{scene_name}_{os.path.basename(lr_img_path)}"))
                        shutil.move(hr_img_path, os.path.join(hr_dir_path, f"{scene_name}_{os.path.basename(hr_img_path)}"))

            # Delete old scene directories
            for scene_dir_path in scene_dir_paths:
                shutil.rmtree(scene_dir_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()

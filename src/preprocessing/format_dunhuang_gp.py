import os
import sys
import ast
import shutil
import argparse
import numpy as np
import monai.transforms as transforms
from glob import glob
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.transforms import EnsureChannelExists, ThresholdIntensity, ReverseDimensions


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

            # Format images
            split = os.path.basename(dir_path)[:-3]

            for src_dir, dest_dir in [(f'{split}_masked', 'LR'), (f'{split}_GT', 'HR')]:
                # Fetch all data file paths
                dataset_file_paths = sorted(file_path for file_path in glob(os.path.join(dir_path, src_dir, '*')) if os.path.basename(file_path) != '.DS_Store')

                for file_path in dataset_file_paths:
                    # Load image
                    img = np.array(Image.open(file_path))
                    min_spatial_dimensions = min(img.shape[:-1])

                    # Uniform shape transforms
                    uniform_transforms = transforms.Compose([
                        ReverseDimensions(),
                        EnsureChannelExists(num_spatial_dimensions=2),
                        transforms.CenterSpatialCrop(roi_size=min_spatial_dimensions),
                        transforms.Resize(spatial_size=args.uniform_shape, mode="bicubic"),
                        ThresholdIntensity(threshold=args.min, above=True, cval=args.min, channel_wise=True),
                        ThresholdIntensity(threshold=args.max, above=False, cval=args.max, channel_wise=True),
                        ReverseDimensions(),
                        transforms.ToNumpy(dtype=np.uint8)
                    ])

                    # Apply transforms
                    img = uniform_transforms(img)

                    # Save the transformed image
                    Image.fromarray(img).save(file_path.replace(src_dir, dest_dir).replace(".jpg", ".png"))

                # Delete old directory
                shutil.rmtree(os.path.join(dir_path, src_dir))

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--min', default=0, help='Minimum pixel value', type=int)
    parser.add_argument('--max', default=255, help='Maximum pixel value', type=int)
    parser.add_argument('--uniform-shape', default=(256, 256), help='Uniform shape for image standardization', type=ast.literal_eval)
    args = parser.parse_args()

    main()

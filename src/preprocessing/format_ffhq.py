import os
import sys
import argparse
import numpy as np
import monai.transforms as transforms
from glob import glob
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.transforms import MoveDimension, ThresholdIntensity


def main():
    try:
        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*.png')))

        # Apply data reduction (subsampling)
        if args.keep_ratio < 1.0:
            number_files_kept = max(1, int(len(dataset_file_paths) * args.keep_ratio))
            keep_files = set(np.random.choice(dataset_file_paths, number_files_kept, replace=False))

            # Delete remaining files
            for file_path in dataset_file_paths:
                if file_path not in keep_files:
                    os.remove(file_path)

            # Update list to only kept files
            dataset_file_paths = sorted(list(keep_files))

        # Uniform shape transforms
        uniform_transforms = transforms.Compose([
            MoveDimension(source=-1, destination=0),
            transforms.Zoom(zoom=1/args.scale_factor, mode="bicubic", keep_size=False),
            ThresholdIntensity(threshold=args.min, above=True, cval=args.min, channel_wise=True),
            ThresholdIntensity(threshold=args.max, above=False, cval=args.max, channel_wise=True),
            MoveDimension(source=0, destination=-1),
            transforms.ToNumpy()
        ])

        for file_path in dataset_file_paths:
            # Load image
            img = np.array(Image.open(file_path)).astype(np.float32)

            # Apply the transformations
            transformed_img = uniform_transforms(img)

            # Save the transformed image
            Image.fromarray(transformed_img.astype(np.uint8)).save(file_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--min', default=0, help='Minimum pixel value', type=int)
    parser.add_argument('--max', default=255, help='Maximum pixel value', type=int)
    parser.add_argument('--scale-factor', default=4, help='Scale factor to reduce data resolution and alleviate computation', type=int)
    parser.add_argument('--keep-ratio', default=1.0, help='Fraction of samples kept in data reduction to alleviate computation', type=float)
    args = parser.parse_args()

    main()

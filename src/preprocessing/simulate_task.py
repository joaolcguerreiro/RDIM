import os
import sys
import shutil
import argparse
import numpy as np
import monai.transforms as transforms
import torchvision.transforms as T
from glob import glob
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.transforms import MoveDimension, ThresholdIntensity


def main():
    try:
        # Common transforms
        initial_transforms = transforms.Compose([
            MoveDimension(source=-1, destination=0),
        ])

        final_transforms = transforms.Compose([
            ThresholdIntensity(threshold=args.min, above=True, cval=args.min, channel_wise=True),
            ThresholdIntensity(threshold=args.max, above=False, cval=args.max, channel_wise=True),
            MoveDimension(source=0, destination=-1),
            transforms.ToNumpy()
        ])

        # Task Specific Transforms
        task_transforms = {
            'Inpainting': transforms.Compose([
                transforms.Lambda(lambda x: x * (np.random.rand(*x.shape) > args.mask_prob))
            ]),
            'Colorization': transforms.Compose([
                transforms.ToTensor(),
                T.Grayscale(num_output_channels=1)
            ]),
            'Deblurring': transforms.Compose([
                transforms.ToTensor(),
                T.GaussianBlur(kernel_size=args.blur_size, sigma=args.blur_std)
            ])
        }

        if not os.path.exists(args.dir):
            raise FileNotFoundError(f"Directory with data '{args.dir}' does not exist")

        # Remove and create LR and HR directories
        lr_dir_path = os.path.join(args.dir, 'LR')
        hr_dir_path = os.path.join(args.dir, 'HR')

        if os.path.exists(lr_dir_path):
            shutil.rmtree(lr_dir_path)

        os.makedirs(lr_dir_path)

        if os.path.exists(hr_dir_path):
            shutil.rmtree(hr_dir_path)

        os.makedirs(hr_dir_path)

        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*.png')))

        for file_index, file_path in enumerate(dataset_file_paths):
            # Load image
            img = np.array(Image.open(file_path)).astype(np.uint8)

            # Apply the transformations
            transformed_img = final_transforms(task_transforms[args.task](initial_transforms(img)))

            # Save volumes
            filename = os.path.basename(file_path)

            # Save image pair
            Image.fromarray((transformed_img if transformed_img.shape[-1] > 1 else transformed_img.squeeze(-1)).astype(np.uint8)).save(os.path.join(args.dir, 'LR', filename))
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(args.dir, 'HR', filename))

        # Remove root images
        for file_path in dataset_file_paths:
            os.remove(file_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Simulate Task Specific Corrupted Images')
    parser.add_argument('--dir', required=True, help='Path to directory with data', type=str)
    parser.add_argument('--task', required=True, help='Image restoration task (Inpainting, Colorization, or Deblurring)', type=str)
    parser.add_argument('--min', default=0, help='Minimum pixel value', type=int)
    parser.add_argument('--max', default=255, help='Maximum pixel value', type=int)
    parser.add_argument('--mask-prob', default=0.5, help='Probability of each pixel being masked and set to 0', type=float)
    parser.add_argument('--blur-std', default=3.0, help='Gaussian blur kernel standard deviation', type=float)
    parser.add_argument('--blur-size', default=15, help='Gaussian blur kernel size', type=int)
    args = parser.parse_args()

    main()

import os
import sys
import shutil
import argparse
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from glob import glob
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import EnsureChannelExists, RandGaussianSmooth, RandForegroundGaussianNoise, RandForegroundStreakNoise


def main():
    try:
        # Common transforms
        initial_transforms = transforms.Compose([
            EnsureChannelExists(),
        ])

        final_transforms = transforms.Compose([
            transforms.ToNumpy()
        ])

        # Low-Resolution Transforms
        lr_transforms = transforms.Compose([
            RandGaussianSmooth(prob=0.5, sigma=(0.0, 1.0)),
            transforms.Zoom(zoom=(1 / args.scale_factor), mode="trilinear", keep_size=False) if args.scale_factor else transforms.Identity(),
            RandForegroundGaussianNoise(prob=0.5, mean=0.01, std=0.03),
            RandForegroundStreakNoise(prob=0.5, intensity=1.1, line_thickness=1, num_origins=(40, 90), num_streaks=(60, 120), origin_radius_ratio=1.5, slope_variation=0.7, white_noise=False)
        ])

        # High-Resolution Transforms
        hr_transforms = transforms.Compose([
        ])

        # Remove and create LR and HR directories
        for directory in [args.train_dir, args.val_dir, args.test_dir]:
            if os.path.exists(directory):
                # Remove LR and HR directories if exists
                shutil.rmtree(os.path.join(directory, 'LR')) if os.path.exists(os.path.join(directory, 'LR')) else None
                shutil.rmtree(os.path.join(directory, 'HR')) if os.path.exists(os.path.join(directory, 'HR')) else None

                # Create LR and HR directories
                os.makedirs(os.path.join(directory, 'LR'), exist_ok=True)
                os.makedirs(os.path.join(directory, 'HR'), exist_ok=True)

            # Fetch all data file paths
            data_files = sorted(glob(os.path.join(directory, '*.nii*')))

            for index, file in enumerate(data_files):
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    for augmentation_factor in range(args.augmentation_factor if directory == args.train_dir else 1):
                        # Set transforms seed
                        lr_transforms.set_random_state(seed=index * args.augmentation_factor + augmentation_factor)
                        lr_transforms.set_random_state(seed=index * args.augmentation_factor + augmentation_factor)

                        # Load the NIfTI file
                        nifti_img = nib.load(file)
                        volume = initial_transforms(nifti_img.get_fdata())

                        # Apply the transformations
                        lr_volume = np.around(final_transforms(lr_transforms(volume))).astype(np.int16)
                        hr_volume = np.around(final_transforms(hr_transforms(volume))).astype(np.int16)

                        # Update header of the low-resolution volume
                        new_lr_header = nifti_img.header.copy()
                        new_lr_header.set_zooms(np.array(new_lr_header.get_zooms()) * args.scale_factor)
                        new_lr_header.set_data_shape(lr_volume.shape)

                        # Update the low-resolution affine matrix to account for the new voxel dimensions
                        new_lr_affine = nifti_img.affine.copy()
                        np.fill_diagonal(new_lr_affine[:3, :3], np.diagonal(new_lr_affine[:3, :3]) * args.scale_factor)

                        # Create NIfTI images
                        lr_nifti_img = nib.Nifti1Image(lr_volume, new_lr_affine, new_lr_header)
                        hr_nifti_img = nib.Nifti1Image(hr_volume, nifti_img.affine.copy(), nifti_img.header.copy())

                        # Save volumes
                        nib.save(lr_nifti_img, os.path.join(directory, 'LR', f'LIDC-IDRI-{index * args.augmentation_factor + augmentation_factor:05}'))
                        nib.save(hr_nifti_img, os.path.join(directory, 'HR', f'LIDC-IDRI-{index * args.augmentation_factor + augmentation_factor:05}'))

            # Remove old files
            for file in data_files:
                if (file.endswith('.nii') or file.endswith(".nii.gz")) and os.path.exists(file):
                    os.remove(file)
    except Exception as e:
        print("Check directory paths. If error remains split or download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Downsample data')
    parser.add_argument('--train-dir', default=os.path.join('..', '..', 'data', 'training'), help='Path to destination training directory', type=str)
    parser.add_argument('--val-dir', default=os.path.join('..', '..', 'data', 'validation'), help='Path to destination validation directory', type=str)
    parser.add_argument('--test-dir', default=os.path.join('..', '..', 'data', 'testing'), help='Path to destination testing directory', type=str)
    parser.add_argument('--scale-factor', default=2, help='Resolution ratio between LR and HR', type=int)
    parser.add_argument('--augmentation-factor', default=2, help='Factor to define how many lr-hr pairs will be created for each data sample', type=int)
    args = parser.parse_args()

    main()

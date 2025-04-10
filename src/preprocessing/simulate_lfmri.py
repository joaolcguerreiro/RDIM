import re
import os
import sys
import shutil
import argparse
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.transforms import RemoveChannelDimension, EnsureChannelExists, ReverseDimensions, RandSimulateLowField


def main():
    try:
        # Common transforms
        initial_transforms = transforms.Compose([
            ReverseDimensions(),
            RemoveChannelDimension(num_spatial_dimensions=3),
        ])

        final_transforms = transforms.Compose([
            EnsureChannelExists(num_spatial_dimensions=3),
            ReverseDimensions(),
            transforms.ToNumpy()
        ])

        # Low-Resolution Transforms
        lr_transforms = transforms.Compose([
            RandSimulateLowField(prob=1.0, scale_factor=args.scale_factor, gamma=(args.gamma, args.gamma), std=(args.noise_std, args.noise_std), clamp=True),
        ])

        # High-Resolution Transforms
        hr_transforms = transforms.Compose([
        ])

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
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*.nii.gz')))

        for file_index, file_path in enumerate(dataset_file_paths):
            for augmentation_index in range(args.augmentation_factor):
                # Set transforms seed
                lr_transforms.set_random_state(seed=augmentation_index + file_index * args.augmentation_factor)
                hr_transforms.set_random_state(seed=augmentation_index + file_index * args.augmentation_factor)

                # Load the NIfTI file
                nifti_img = nib.load(file_path)
                volume = initial_transforms(nifti_img.get_fdata())

                # Apply the transformations
                lr_volume = np.around(final_transforms(lr_transforms(volume)))
                hr_volume = np.around(final_transforms(hr_transforms(volume)))

                # Create copies of the header and affine matrix
                lr_header = nifti_img.header.copy()
                lr_affine = nifti_img.affine.copy()

                # Update the header to account for the new voxel size
                lr_header.set_data_shape(lr_volume.shape)
                lr_header.set_zooms(list(np.array(lr_header.get_zooms()) * args.scale_factor)[:3] + list(lr_header.get_zooms())[3:])

                # Create NIfTI images
                lr_nifti_img = nib.Nifti1Image(lr_volume, lr_affine, lr_header)
                hr_nifti_img = nib.Nifti1Image(hr_volume, nifti_img.affine.copy(), nifti_img.header.copy())

                # Save volumes
                filename, file_extension = re.match(r'(.*?)(\.[^.]+(?:\.[^.]+)*)?$', os.path.basename(file_path)).groups()

                nib.save(lr_nifti_img, os.path.join(args.dir, 'LR', filename + f'-{augmentation_index}' + file_extension))
                nib.save(hr_nifti_img, os.path.join(args.dir, 'HR', filename + f'-{augmentation_index}' + file_extension))

        # Remove root NIfTI files
        for file_path in dataset_file_paths:
            os.remove(file_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Simulate Low-Dode CT')
    parser.add_argument('--dir', required=True, help='Path to directory with data', type=str)
    parser.add_argument('--scale-factor', default=2, help='Resolution ratio between LR and HR', type=int)
    parser.add_argument('--gamma', default=0.6, help='Gamma factor to adjust contrast', type=float)
    parser.add_argument('--noise-std', default=2.0, help='Standard deviation of noise', type=float)
    parser.add_argument('--augmentation-factor', default=1, help='Define number of LR-HR pairs created for each data sample', type=int)
    args = parser.parse_args()

    main()

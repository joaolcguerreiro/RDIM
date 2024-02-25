import re
import os
import sys
import shutil
import argparse
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from glob import glob
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import RemoveChannelDimension, EnsureChannelExists, ReverseDimensions, RandSimulateLowDose


def main():
    try:
        # Common transforms
        initial_transforms = transforms.Compose([
            ReverseDimensions(),
            RemoveChannelDimension(),
        ])

        final_transforms = transforms.Compose([
            EnsureChannelExists(),
            transforms.ThresholdIntensity(threshold=args.min, above=True, cval=args.min),
            transforms.ThresholdIntensity(threshold=args.max, above=False, cval=args.max),
            ReverseDimensions(),
            transforms.ToNumpy()
        ])

        # Low-Resolution Transforms
        lr_transforms = transforms.Compose([
            RandSimulateLowDose(prob=1.0, incident_flux_level=(2000, 5000), std=(2.5, 2.5), clamp=True),
        ], map_items=False, unpack_items=True)

        # High-Resolution Transforms
        hr_transforms = transforms.Compose([
        ])

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
                lr_transforms.set_random_state(seed=augmentation_index + file_index * args.augmentation_factor)

                # Load the NIfTI file
                nifti_img = nib.load(file_path)
                volume = initial_transforms(nifti_img.get_fdata())

                # Compute voxel size in meters
                voxel_size = sum(nifti_img.header.get_zooms()[:2]) / len(nifti_img.header.get_zooms()[:2]) / 1000 / args.scale_factor

                # Apply the transformations
                lr_volume = np.around(final_transforms(lr_transforms((volume, voxel_size)))).astype(np.int16)
                hr_volume = np.around(final_transforms(hr_transforms(volume))).astype(np.int16)

                # Create NIfTI images
                lr_nifti_img = nib.Nifti1Image(lr_volume, nifti_img.affine.copy(), nifti_img.header.copy())
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
    parser.add_argument('--min', default=-1000, help='Minimum HU scale value', type=int)
    parser.add_argument('--max', default=1000, help='Maximum HU scale value', type=int)
    parser.add_argument('--augmentation-factor', default=1, help='Define number of LR-HR pairs created for each data sample', type=int)
    parser.add_argument('--scale-factor', default=2, help='Scale factor used to reduce data resolution and alleviate computation', type=int)
    args = parser.parse_args()

    main()

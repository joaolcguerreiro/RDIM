import os
import sys
import ast
import argparse
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.transforms import EnsureChannelExists, ReverseDimensions


def main():
    try:
        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*.nii.gz')))

        # Uniform shape transforms
        uniform_transforms = transforms.Compose([
            ReverseDimensions(),
            EnsureChannelExists(num_spatial_dimensions=3),
            transforms.ScaleIntensity(),
            transforms.SpatialPad(spatial_size=args.uniform_shape),
            transforms.CenterSpatialCrop(roi_size=args.uniform_shape),
            transforms.Zoom(zoom=1/args.scale_factor, mode="trilinear", keep_size=False),
            transforms.ScaleIntensity(minv=args.min, maxv=args.max),
            ReverseDimensions(),
            transforms.ToNumpy()
        ])

        for file_path in dataset_file_paths:
            # Load the NIfTI file
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()

            # Apply the transformations
            uniform_data = np.around(uniform_transforms(data))

            # Create copies of the header and affine matrix
            new_header = nifti_img.header.copy()
            new_affine = nifti_img.affine.copy()

            # Update the header to account for the new voxel size
            new_header.set_data_shape(uniform_data.shape)
            new_header.set_zooms(list(np.array(new_header.get_zooms()) * args.scale_factor)[:3] + list(new_header.get_zooms())[3:])

            # Create a new NIfTI image
            uniform_nifti_img = nib.Nifti1Image(uniform_data, new_affine, new_header)

            # Save the transformed NIfTI file
            nib.save(uniform_nifti_img, file_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--min', default=0, help='Minimum pixel value', type=int)
    parser.add_argument('--max', default=65535, help='Maximum pixel value', type=int)
    parser.add_argument('--uniform-shape', default=(128, 192, 192), help='Uniform shape for volume standardization', type=ast.literal_eval)
    parser.add_argument('--scale-factor', default=1, help='Scale factor to reduce data resolution and alleviate computation', type=int)
    args = parser.parse_args()

    main()

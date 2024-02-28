import os
import sys
import ast
import shutil
import argparse
import pydicom
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from glob import glob
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import EnsureChannelExists, ReverseDimensions


def main():
    try:
        # Define training, validation and testing directories
        train_dir_path = os.path.join(args.dir, 'training')
        val_dir_path = os.path.join(args.dir, 'validation')
        test_dir_path = os.path.join(args.dir, 'testing')

        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
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

            # Transform DICOM to NIfTI file format
            for volume_dir_path in dataset_file_paths:
                for file_path in sorted(glob(os.path.join(volume_dir_path, '*'))):
                    # Load all DICOM files
                    dicom_files = sorted([os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.dcm')])

                    # Load the DICOM files into a list of pydicom datasets
                    datasets = [pydicom.dcmread(file) for file in dicom_files]

                    # Get the voxel dimensions
                    pixel_spacing = datasets[0].PixelSpacing if hasattr(datasets[0], 'PixelSpacing') else [1.0, 1.0]
                    slice_thickness = datasets[0].SliceThickness if hasattr(datasets[0], 'SliceThickness') else 1.0
                    voxel_dims = (pixel_spacing[0], pixel_spacing[1], slice_thickness)

                    # Get the data as a 3D numpy array
                    data = []

                    for dataset in datasets:
                        # Convert to HU scale
                        data.append(dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept)

                    data = np.array(data).astype(np.int16)

                    # Reverse volume dimension order
                    data = data.transpose()

                    # Create a NIfTI image from the DICOM datasets
                    nifti_img = nib.Nifti1Image(data, np.eye(4), None)
                    nifti_img.header.set_zooms(voxel_dims)
                    nifti_img.header.set_data_shape(data.shape)

                    # Save the NIFTI image to a file
                    nib.save(nifti_img, os.path.join(lr_dir_path if 'Low Dose'.lower() in os.path.basename(file_path).lower() else hr_dir_path, os.path.basename(volume_dir_path) + '-raw.nii.gz'))

        # Compute training data range to perform transforms accordingly
        _min = float('+inf')
        _max = float('-inf')

        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            hr_dir_path = os.path.join(dir_path, 'HR')

            # Fetch all data file paths
            dataset_file_paths = sorted(glob(os.path.join(hr_dir_path, '*-raw.nii.gz')))

            for file_path in dataset_file_paths:
                # Load the NIfTI file
                nifti_img = nib.load(file_path)
                data = nifti_img.get_fdata()

                # Compute the minimum and maximum values
                _min = min(_min, data.min())
                _max = max(_max, data.max())

        # Uniform shape transforms
        uniform_transforms = transforms.Compose([
            ReverseDimensions(),
            EnsureChannelExists(),
            transforms.NormalizeIntensity(subtrahend=_min, divisor=(_max-_min)),
            transforms.SpatialPad(spatial_size=args.uniform_shape),
            transforms.CenterSpatialCrop(roi_size=args.uniform_shape),
            transforms.Zoom(zoom=1/args.scale_factor, mode="trilinear", keep_size=False),
            transforms.NormalizeIntensity(subtrahend=-_min/(_max-_min), divisor=1/(_max-_min)),
            transforms.ThresholdIntensity(threshold=args.min, above=True, cval=args.min),
            transforms.ThresholdIntensity(threshold=args.max, above=False, cval=args.max),
            ReverseDimensions(),
            transforms.ToNumpy()
        ])

        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            lr_dir_path = os.path.join(dir_path, 'LR')
            hr_dir_path = os.path.join(dir_path, 'HR')

            for child_dir_path in [lr_dir_path, hr_dir_path]:
                # Fetch all data file paths
                dataset_file_paths = sorted(glob(os.path.join(child_dir_path, '*-raw.nii.gz')))

                for file_path in dataset_file_paths:
                    # Load the NIfTI file
                    nifti_img = nib.load(file_path)
                    data = nifti_img.get_fdata()

                    # Apply the transformations
                    uniform_data = np.around(uniform_transforms(data)).astype(np.int16)

                    # Create copies of the header and affine matrix
                    new_header = nifti_img.header.copy()
                    new_affine = nifti_img.affine.copy()

                    # Update the header to account for the new voxel size
                    new_header.set_data_shape(uniform_data.shape)
                    new_header.set_zooms(list(np.array(new_header.get_zooms()) * args.scale_factor)[:3] + list(new_header.get_zooms())[3:])

                    # Create a new NIfTI image
                    uniform_nifti_img = nib.Nifti1Image(uniform_data, new_affine, new_header)

                    # Save the transformed NIfTI file
                    nib.save(uniform_nifti_img, file_path.replace('-raw', ''))

        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            lr_dir_path = os.path.join(dir_path, 'LR')
            hr_dir_path = os.path.join(dir_path, 'HR')

            for child_dir_path in [lr_dir_path, hr_dir_path]:
                # Fetch all data file paths
                dataset_file_paths = sorted(glob(os.path.join(child_dir_path, '*-raw.nii.gz')))

                for file_path in dataset_file_paths:
                    os.remove(file_path)

        for dir_path in [train_dir_path, val_dir_path, test_dir_path]:
            # Fetch all data file paths
            dataset_file_paths = sorted(file_path for file_path in glob(os.path.join(dir_path, '*')) if os.path.basename(file_path) not in ['.DS_Store', 'LR', 'HR'])

            # Transform DICOM to NIfTI file format
            for volume_dir_path in dataset_file_paths:
                # Remove old directory
                shutil.rmtree(volume_dir_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--min', default=-1000, help='Minimum HU scale value', type=int)
    parser.add_argument('--max', default=1000, help='Maximum HU scale value', type=int)
    parser.add_argument('--uniform-shape', default=(512, 512, 512), help='Uniform shape for volume standardization', type=ast.literal_eval)
    parser.add_argument('--scale-factor', default=2, help='Scale factor to reduce data resolution and alleviate computation', type=int)
    args = parser.parse_args()

    main()

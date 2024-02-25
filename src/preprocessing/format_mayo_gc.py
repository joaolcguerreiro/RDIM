import os
import sys
import ast
import shutil
import zipfile
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
        # Define old training and testing directories
        old_train_dir_path = os.path.join(args.dir, 'Patient_Data', 'Training_Image_Data')
        old_test_dir_path = os.path.join(args.dir, 'Patient_Data', 'Testing_Image_Data')

        # Rename training and testing directories
        train_dir_path = os.path.join(args.dir, 'training')
        test_dir_path = os.path.join(args.dir, 'testing')

        for new_dir_path, old_dir_path in zip([train_dir_path, test_dir_path], [old_train_dir_path, old_test_dir_path]):
            if os.path.exists(old_dir_path):
                os.rename(old_dir_path, new_dir_path)

                # Create LR and HR directories
                os.makedirs(os.path.join(new_dir_path, 'LR'))
                os.makedirs(os.path.join(new_dir_path, 'HR'))

        # Unzip files
        for dir_path in [train_dir_path, test_dir_path]:
            # Fetch all data directory paths
            child_dir_paths = sorted(glob(os.path.join(dir_path, '*')))

            for child_dir_path in child_dir_paths:
                # Pass LR and HR directories
                if os.path.basename(child_dir_path) in ['LR', 'HR']:
                    continue

                # Fetch all data directory paths
                zip_file_paths = sorted(glob(os.path.join(child_dir_path, '*.zip')))

                for zip_file_path in zip_file_paths:
                    # Unzip file
                    zip_filename = os.path.basename(zip_file_path)

                    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                        zip_file.extractall(os.path.join(dir_path, 'LR' if zip_filename.startswith('QD_') or zip_filename.startswith('quarter_') else 'HR'))

                    # Delete zip file
                    os.remove(zip_file_path)

                    # Rename file
                    file_dir_path = os.path.join(dir_path, 'LR' if zip_filename.startswith('QD_') else 'HR')
                    os.rename(os.path.join(file_dir_path, zip_filename), os.path.join(file_dir_path, os.path.basename(child_dir_path)))

        # Transform IMA to NIfTI file format
        for dir_path in [os.path.join(split_dir_path, kind) for split_dir_path in [train_dir_path, test_dir_path] for kind in ['LR', 'HR']]:
            # Fetch all data directory paths
            child_dir_paths = sorted(glob(os.path.join(dir_path, '*')))

            for child_dir_path in child_dir_paths:
                # Pass non-directory files
                if not os.path.isdir(child_dir_path):
                    continue

                # Fetch all data directory paths
                dataset_file_paths = sorted(glob(os.path.join(child_dir_path, '*')))

                for volume_dir_path in dataset_file_paths:
                    volume_type = os.path.basename(child_dir_path)
                    volume_name = os.path.basename(volume_dir_path) + '_' + volume_type.replace(' ', '_')

                    # Define redundant directory path
                    redundant_dir_path = os.path.join(volume_dir_path, os.listdir(volume_dir_path)[0])

                    # Load all IMA files
                    ima_files = sorted([os.path.join(redundant_dir_path, file) for file in os.listdir(redundant_dir_path) if file.endswith('.IMA')])

                    # Load the IMA files into a list of ima datasets
                    datasets = [pydicom.dcmread(file) for file in ima_files]

                    # Get the voxel dimensions
                    pixel_spacing = datasets[0].PixelSpacing
                    slice_thickness = datasets[0].SliceThickness
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
                    nib.save(nifti_img, os.path.join(dir_path, volume_name) + "-raw.nii.gz")

                    # Remove old directory
                    shutil.rmtree(volume_dir_path)

        # Uniform shape transforms
        uniform_transforms = transforms.Compose([
            ReverseDimensions(),
            EnsureChannelExists(),
            transforms.SpatialPad(spatial_size=args.uniform_shape),
            transforms.CenterSpatialCrop(roi_size=args.uniform_shape),
            transforms.Zoom(zoom=(1 / args.scale_factor), mode="trilinear", keep_size=False),
            ReverseDimensions(),
            transforms.ToNumpy()
        ])

        for dir_path in [os.path.join(split_dir_path, kind) for split_dir_path in [train_dir_path, test_dir_path] for kind in ['LR', 'HR']]:
            # Fetch all data file paths
            dataset_file_paths = sorted(glob(os.path.join(dir_path, '*-raw.nii.gz')))

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

            # Remove raw NIfTI files
            for file_path in dataset_file_paths:
                os.remove(file_path)

        # Remove old directories
        for dir_path in [train_dir_path, test_dir_path]:
            # Fetch all data directory paths
            child_dir_paths = sorted(glob(os.path.join(dir_path, '*')))

            for child_dir_path in child_dir_paths:
                # Pass LR and HR directories
                if os.path.basename(child_dir_path) in ['LR', 'HR']:
                    continue

                shutil.rmtree(child_dir_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--uniform-shape', default=(-1, -1, -1), help='Uniform shape for volume standardization', type=ast.literal_eval)
    parser.add_argument('--scale-factor', default=1, help='Scale factor to reduce data resolution and alleviate computation', type=int)
    args = parser.parse_args()

    main()

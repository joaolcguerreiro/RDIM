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
        # Move the data from the redundant intermediary directory to the root directory
        src_dir_path = os.path.join(args.dir, 'LIDC-IDRI')
        dest_dir_path = args.dir

        if os.path.exists(src_dir_path):
            for file_path in os.listdir(src_dir_path):
                if file_path != ".DS_Store":
                    shutil.move(os.path.join(src_dir_path, file_path), dest_dir_path)

            # Remove redundant intermediary directory
            shutil.rmtree(src_dir_path)

        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*')))

        # Simplify dataset hierarchy by removing redundant intermediary folders
        for _ in range(2):
            for parent_dir_path in dataset_file_paths:
                # Pass non-directory files
                if not os.path.isdir(parent_dir_path):
                    continue

                # Check if redundant folder was already removed
                if len([child_file_path for child_file_path in os.listdir(parent_dir_path) if os.path.isdir(os.path.join(parent_dir_path, child_file_path))]) == 0:
                    continue

                # Get path of the child directory
                child_dir_path = os.path.join(parent_dir_path, [child_file_path for child_file_path in os.listdir(parent_dir_path) if child_file_path != ".DS_Store" and os.path.isdir(os.path.join(parent_dir_path, child_file_path))][0])

                # Move the data from the child directory to the parent directory
                for file_path in os.listdir(child_dir_path):
                    if file_path != ".DS_Store":
                        shutil.move(os.path.join(child_dir_path, file_path), parent_dir_path)

                # Remove redundant child directory
                shutil.rmtree(child_dir_path)

        # Transform DICOM to NIfTI file format
        for volume_dir_path in dataset_file_paths:
            # Pass non-directory files
            if not os.path.isdir(volume_dir_path):
                continue

            # Remove scans with less than minimum required slices
            if len(os.listdir(volume_dir_path)) - 1 < args.min_slices:
                shutil.rmtree(volume_dir_path)
            else:
                # Load all DICOM files
                dicom_files = sorted([os.path.join(volume_dir_path, file) for file in os.listdir(volume_dir_path) if file.endswith('.dcm')])

                # Load the DICOM files into a list of pydicom datasets
                datasets = [pydicom.dcmread(file) for file in dicom_files]

                # Get the voxel dimensions
                pixel_spacing = datasets[0].PixelSpacing
                slice_thickness = datasets[0].SliceThickness
                voxel_dims = (pixel_spacing[0], pixel_spacing[1], slice_thickness)

                # Check slice thickness
                if slice_thickness > args.max_slice_thick:
                    shutil.rmtree(volume_dir_path)
                    continue

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
                nib.save(nifti_img, volume_dir_path + "-raw.nii.gz")

                # Remove old directory
                shutil.rmtree(volume_dir_path)

        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*-raw.nii.gz')))

        # Compute training data range to perform transforms accordingly
        _min = float('+inf')
        _max = float('-inf')

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
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--min-slices', default=256, help='Minimum number of slices per volume', type=int)
    parser.add_argument('--max-slice-thick', default=1.25, help='Maximum slice thickness per volume', type=float)
    parser.add_argument('--min', default=-1024, help='Minimum HU scale value', type=int)
    parser.add_argument('--max', default=3071, help='Maximum HU scale value', type=int)
    parser.add_argument('--uniform-shape', default=(512, 512, 512), help='Uniform shape for volume standardization', type=ast.literal_eval)
    parser.add_argument('--scale-factor', default=2, help='Scale factor to reduce data resolution and alleviate computation', type=int)
    args = parser.parse_args()

    main()

import shutil
import os
import ast
import argparse
import pydicom
import numpy as np
import nibabel as nib
import monai.transforms as transforms
from glob import glob


def main():
    try:
        if args.old_dir:
            # Remove metadata and license files
            license_file = os.path.join(args.old_dir, 'LIDC-IDRI', 'LICENSE')
            metadata_file = os.path.join(args.old_dir, 'metadata.csv')

            if os.path.exists(license_file):
                os.remove(license_file)

            if os.path.exists(metadata_file):
                os.remove(metadata_file)

            if os.path.exists(args.new_dir):
                shutil.rmtree(args.new_dir)

            # Remove redundant intermediary folder
            if os.path.exists(args.old_dir):
                src = os.path.join(args.old_dir, 'LIDC-IDRI')
                dst = args.new_dir
                shutil.move(src, dst)
                shutil.rmtree(args.old_dir)

        # Fetch all data file paths
        dataset_files = sorted(glob(os.path.join(args.new_dir, '*')))

        # Simplify dataset hierarchy by removing redundant intermediary folders
        for _ in range(2):
            for parent_dir in dataset_files:
                # Pass non-directory files
                if not os.path.isdir(parent_dir):
                    continue

                # Check if redundant folder was already removed
                if len([child_file for child_file in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, child_file))]) == 0:
                    continue

                # Get the path of the child directory
                child_dir = os.path.join(parent_dir, [file for file in os.listdir(parent_dir) if file != ".DS_Store" and os.path.isdir(os.path.join(parent_dir, file))][0])

                # Move all the contents of the child directory to the parent directory
                for file in os.listdir(child_dir):
                    src = os.path.join(child_dir, file)
                    dst = os.path.join(parent_dir, file)
                    shutil.move(src, dst)

                # Remove the child directory
                shutil.rmtree(child_dir)

        # Transform DICOM to NIfTI file format
        for volume_dir in dataset_files:
            # Pass non-directory files
            if not os.path.isdir(volume_dir):
                continue

            # Remove scans with less than minimum required slices
            if len(os.listdir(volume_dir)) - 1 < args.min_slices:
                shutil.rmtree(volume_dir)
            else:
                # Load all DICOM files
                dicom_files = sorted([os.path.join(volume_dir, file) for file in os.listdir(volume_dir) if file.endswith('.dcm')])

                # Load the DICOM files into a list of pydicom datasets
                datasets = [pydicom.dcmread(file) for file in dicom_files]

                # Get the voxel dimensions
                pixel_spacing = datasets[0].PixelSpacing
                slice_thickness = datasets[0].SliceThickness
                voxel_dims = (pixel_spacing[0], pixel_spacing[1], slice_thickness)

                # Check slice thickness
                if slice_thickness > args.min_slice_thick:
                    shutil.rmtree(volume_dir)
                    continue

                # Get the data as a 3D numpy array
                data = []

                for dataset in datasets:
                    # Convert to HU scale
                    data.append(dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept)

                data = np.array(data).astype(np.int16)

                # Create a NIfTI image from the DICOM datasets
                nifti_img = nib.Nifti1Image(data, np.eye(4), None)
                nifti_img.header.set_zooms(voxel_dims)

                # Save the NIFTI image to a file
                nib.save(nifti_img, volume_dir + ".nii.gz")

        # Fetch all data file paths
        dataset_files = sorted(glob(os.path.join(args.new_dir, '*')))

        # Compute training data range to perform transforms accordingly
        _min = float('+inf')
        _max = float('-inf')

        for file in dataset_files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # Load the NIfTI file
                nifti_img = nib.load(file)
                data = nifti_img.get_fdata()

                # Compute the minimum and maximum values
                _min = min(_min, data.min())
                _max = max(_max, data.max())

        # Uniform shape transforms
        uniform_transforms = transforms.Compose([
            transforms.NormalizeIntensity(subtrahend=_min, divisor=(_max-_min)),
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.SpatialPad(spatial_size=args.uniform_shape),
            transforms.CenterSpatialCrop(roi_size=args.uniform_shape),
            transforms.Zoom(zoom=(1 / args.scale_factor), mode="trilinear", keep_size=False),
            transforms.NormalizeIntensity(subtrahend=-_min/(_max-_min), divisor=1/(_max-_min)),
            transforms.SqueezeDim(),
            transforms.ToNumpy()
        ])

        for file in dataset_files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # Load the NIfTI file
                nifti_img = nib.load(file)
                data = nifti_img.get_fdata()

                # Apply the transformations
                uniform_data = np.around(uniform_transforms(data)).astype(np.int16)

                # Update header of the transformed image
                new_header = nifti_img.header.copy()
                new_header.set_zooms(np.array(new_header.get_zooms()) * args.scale_factor)
                new_header.set_data_shape(uniform_data.shape)

                # Update the affine matrix to account for the new voxel dimensions
                new_affine = nifti_img.affine.copy()
                np.fill_diagonal(new_affine[:3, :3], np.diagonal(new_affine[:3, :3]) * args.scale_factor)

                # Create a new NIfTI image
                uniform_nifti_img = nib.Nifti1Image(uniform_data, new_affine, new_header)

                # Save the transformed NIfTI file
                nib.save(uniform_nifti_img, file)

        # Remove old directories
        for volume_dir in dataset_files:
            if not (file.endswith('.nii') or volume_dir.endswith(".nii.gz")) and os.path.exists(volume_dir):
                shutil.rmtree(volume_dir)
    except Exception as e:
        print("Check directory paths. If error remains download data again. An error occurred: {}".format(e))


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--old-dir', default=max((path for path in glob(os.path.join('..', '..', 'data', 'manifest*')) if os.path.isdir(path)), key=os.path.getmtime, default=None), help='Path to directory with data', type=str)
    parser.add_argument('--new-dir', default=os.path.join('..', '..', 'data', 'LIDC-IDRI'), help='Path to destiny directory', type=str)
    parser.add_argument('--min-slices', default=256, help='Minimum number of slices per volume', type=int)
    parser.add_argument('--min-slice-thick', default=1.25, help='Minimum slice thickness per volume', type=float)
    parser.add_argument('--uniform-shape', default=(512, 512, 512), help='Uniform shape for volume standardization', type=ast.literal_eval)
    parser.add_argument('--scale-factor', default=2, help='Scale factor to reduce data resolution and alleviate computation', type=int)
    args = parser.parse_args()

    main()

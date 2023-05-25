import shutil
import os
import argparse
import numpy as np
import pydicom
import nibabel as nib
from glob import glob


def main():
    # Data path
    old_dir_path = max(glob(os.path.join('..', '..', 'data', 'manifest*')), key=os.path.getmtime)
    new_dir_path = os.path.join('..', '..', 'data', 'LIDC-IDRI')

    # Remove metadata and license files
    license_file = os.path.join(old_dir_path, 'LIDC-IDRI', 'LICENSE')
    metadata_file = os.path.join(old_dir_path, 'metadata.csv')

    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    if os.path.exists(license_file):
        os.remove(license_file)

    # Remove redundant intermediary folder
    if os.path.exists(old_dir_path):
        src = os.path.join(old_dir_path, 'LIDC-IDRI')
        dst = new_dir_path
        shutil.move(src, dst)
        shutil.rmtree(old_dir_path)

    # Fetch all training data file paths
    dataset_files = sorted(glob(os.path.join(new_dir_path, '*')))

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
        if len(os.listdir(volume_dir)) - 1 < args.slices:
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
            if slice_thickness > 1.25:
                shutil.rmtree(volume_dir)
                continue

            # Get the data as a 3D numpy array
            data = []
            for dataset in datasets:
                data.append(dataset.pixel_array)
            data = np.array(data)

            # Create a NIfTI image from the DICOM datasets
            nifti_img = nib.Nifti1Image(data, np.eye(4), None)
            nifti_img.header.set_zooms(voxel_dims)

            # Save the NIFTI image to a file
            nib.save(nifti_img, volume_dir + ".nii.gz")

    # Remove old directories
    for volume_dir in dataset_files:
        if not volume_dir.endswith(".nii.gz") and os.path.exists(volume_dir):
            shutil.rmtree(volume_dir)


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Split training data')
    parser.add_argument('--slices', default=256, help='Minimum number of slices per volume', type=int)
    args = parser.parse_args()

    main()

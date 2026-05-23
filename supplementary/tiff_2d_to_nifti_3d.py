import re
import os
import sys
import shutil
import argparse
import nibabel as nib
import numpy as np
import tifffile
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.transforms import ReverseDimensions, MoveDimension


def natural_sort_key(file_path):
    """Key function for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(file_path))]


def main():
    if os.path.exists(args.nifti_dir):
        shutil.rmtree(args.nifti_dir)

    os.makedirs(args.nifti_dir)

    # Fetch all data file paths
    tiff_file_paths = sorted(glob(os.path.join(args.tiff_dir, '*')))
    sample_ids = set()

    # Collect sample IDs based on filename prefix before first underscore
    for file_path in tiff_file_paths:
        sample_id = os.path.basename(file_path).split('_')[0]
        sample_ids.add(sample_id)

    for sample_id in sample_ids:
        # Fetch all data file paths
        tiff_file_paths = sorted(glob(os.path.join(args.tiff_dir, f'{sample_id}_*.tiff')), key=natural_sort_key)

        # Load TIFF slices
        slices = [tifffile.imread(tiff_file_path).astype(np.float32) for tiff_file_path in tiff_file_paths]

        # Stack into a 3D or 4D array
        data = np.stack(slices, axis=0)

        if data.ndim == 4:
            data = MoveDimension(source=-1, destination=0)(data)
        elif data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        else:
            raise NotImplementedError(f"Unexpected data dimensions ({data.ndim})")

        data = ReverseDimensions()(data)

        # Save as NIfTI files
        nifti_file = nib.Nifti1Image(data.numpy().astype(np.float32), np.eye(4))
        nib.save(nifti_file, os.path.join(args.nifti_dir, f'{sample_id}.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Formatting Script')
    parser.add_argument('--nifti-dir', required=True, help='Path to directory with nifti datas', type=str)
    parser.add_argument('--tiff-dir', required=True, help='Path to directory with numpy data', type=str)
    args = parser.parse_args()

    main()

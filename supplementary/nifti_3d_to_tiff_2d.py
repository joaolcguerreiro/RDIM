import os
import sys
import shutil
import argparse
import nibabel as nib
import tifffile
import numpy as np
import monai.transforms as transforms
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.transforms import RemoveChannelDimension, ReverseDimensions, MoveDimension


def main():
    if os.path.exists(args.tiff_dir):
        shutil.rmtree(args.tiff_dir)

    os.makedirs(args.tiff_dir)

    if not args.same_dir:
        os.makedirs(os.path.join(args.tiff_dir, 'HR'))
        os.makedirs(os.path.join(args.tiff_dir, 'LR'))

    # Fetch all data file paths
    hq_nifti_file_paths = sorted(glob(os.path.join(args.nifti_dir, 'HR', '*')))
    lq_nifti_file_paths = sorted(glob(os.path.join(args.nifti_dir, 'LR', '*')))

    for hq_nifti_file_path, lq_nifti_file_path in zip(hq_nifti_file_paths, lq_nifti_file_paths):
        # Load nifti and format data
        hq_data = ReverseDimensions()(nib.load(hq_nifti_file_path).get_fdata().astype(np.float32))
        lq_data = ReverseDimensions()(nib.load(lq_nifti_file_path).get_fdata().astype(np.float32))

        if args.scale_factor > 1:
            lq_data = transforms.Zoom(zoom=(args.scale_factor, 1, 1), mode="trilinear", keep_size=False)(lq_data)

            assert hq_data.shape[1] == lq_data.shape[1], f"The depth dimension of lq volume ({lq_data.shape[1]}) does not match the depth dimension of hq volume ({hq_data.shape[1]})"

        if not args.channel:
            hq_data = RemoveChannelDimension(num_spatial_dimensions=3)(hq_data)
            lq_data = RemoveChannelDimension(num_spatial_dimensions=3)(lq_data)
        else:
            hq_data = MoveDimension(source=0, destination=-1)(hq_data)
            lq_data = MoveDimension(source=0, destination=-1)(lq_data)

        # Save each slice as TIFF
        for slice_index in range(hq_data.shape[0]):
            hq_slice = hq_data[slice_index, ...]
            lq_slice = lq_data[slice_index, ...]

            filename = os.path.basename(hq_nifti_file_path).split('.')[0]

            if not args.same_dir:
                tifffile.imwrite(os.path.join(args.tiff_dir, 'HR', f"{filename}_{slice_index}.tiff"), hq_slice.astype(np.float32))
                tifffile.imwrite(os.path.join(args.tiff_dir, 'LR', f"{filename}_{slice_index}.tiff"), lq_slice.astype(np.float32))
            else:
                tifffile.imwrite(os.path.join(args.tiff_dir, f"{filename}_{slice_index}_target.tiff"), hq_slice.astype(np.float32))
                tifffile.imwrite(os.path.join(args.tiff_dir, f"{filename}_{slice_index}_input.tiff"), lq_slice.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIfTI 3D to TIFF 2D Conversion Script')
    parser.add_argument('--nifti-dir', required=True, help='Path to directory with NIfTI data', type=str)
    parser.add_argument('--tiff-dir', required=True, help='Path to directory for TIFF output', type=str)
    parser.add_argument('--scale-factor', default=1, help='Resolution ratio between LR and HR', type=int)
    parser.add_argument('--channel', default=True, help='Keep channel dimension', type=(lambda arg: (str(arg).lower() == "true")))
    parser.add_argument('--same-dir', default=False, help='Save input/target pairs in the same directory', type=(lambda arg: (str(arg).lower() == "true")))
    args = parser.parse_args()

    main()

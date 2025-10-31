import os
import re
import argparse
import numpy as np
import nibabel as nib
from PIL import Image


def main():
    try:
        # Validate file extension
        file_extension = (re.search(r'(\.[^.]+(?:\.[^.]+)*)$', os.path.basename(args.dir)) or ['']).group(1).lower()

        if file_extension not in ['.nii', '.nii.gz']:
            raise ValueError(f"Unsupported image format: {file_extension}")

        # Load NIfTI volume
        nifti_img = nib.load(args.dir)
        volume = nifti_img.get_fdata()

        # Check that volume is 3D or 4D with channels
        if volume.ndim not in [3, 4]:
            raise ValueError(f"Unsupported volume dimensions: {volume.shape}. Expected 3D or 4D.")

        # Determine save directory
        parent_dir = os.path.dirname(os.path.abspath(args.dir))
        base_name = os.path.splitext(os.path.basename(args.dir))[0]

        # Handle ".nii.gz"
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]

        save_dir = os.path.join(parent_dir, f"{base_name}-2D")

        # Create save directory if not exists
        os.makedirs(save_dir, exist_ok=True)

        # Check that slice dimension is valid
        if args.slice_dim < 0 or args.slice_dim >= volume.ndim:
            raise ValueError(f"Invalid slice_dim: {args.slice_dim}. Volume has {volume.ndim} dimensions.")

        # Iterate over slices
        for slice_index in range(volume.shape[args.slice_dim]):
            # Take one slice along specified dimension
            image = np.take(volume, slice_index, axis=args.slice_dim)

            # Save slice
            filename = f"{base_name}_slice_{slice_index:05d}.png"
            Image.fromarray((image if (image.ndim > 2 and image.shape[-1] > 1) else image.squeeze(-1)).astype(np.uint8)).save(os.path.join(save_dir, filename))

        print(f"Sliced volume saved to: {save_dir}")

    except Exception as e:
        print(f"An error occurred: {e}.\n" + "If error remains download data again.")


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Slice 3D NIfTI volume into 2D images')
    parser.add_argument('--dir', required=True, help='Path to directory with processed 3D data', type=str)
    parser.add_argument('--slice-dim', default=2, help='Dimension along which to slice', type=int)
    args = parser.parse_args()

    main()

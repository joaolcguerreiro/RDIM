import os
import numpy as np
import nibabel as nib
from glob import glob


# Load Modality/Segmentation Volume (3D MRI)
def load_volume(kind, volume_dir):
    return nib.load(glob(volume_dir + f"/*_{kind}.nii.gz")[0])


# Save Modality/Segmentation Volume (3D MRI)
def save_volume(volume, save_dir, file_name, kind, affine, header):
    # Create Save Directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nib.save(
        nib.Nifti1Image(volume, affine, header=header),
        os.path.join(save_dir, file_name + f"_{kind}.nii.gz"),
    )


# Get Modality/Segmentation Volume (3D MRI)
def get_volume_data(kind, volume_dir):
    return load_volume(kind, volume_dir).get_fdata().astype(np.uint8 if kind == 'seg' else np.float32)[:, :, :]


# Get Modality/Segmentation Volume Affine
def get_volume_affine(kind, volume_dir):
    # Orientation of medical volumes: There are three axes ([Right, Left], [Anterior, Posterior], [Inferior, Superior]) equivalent to the 3d coordinate system (Y, X, Z)
    # 1. First voxel axis goes from Right to Left (y-axis)
    # 2. Second voxel axis goes from Posterior to Anterior (x-axis)
    # 3. Third voxel axis goes from Inferior to Superior (z-axis)
    # see http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm for more information

    # BraTS affine is LPS (left, posterior, superior) with dimensions (Y, X, Z)
    return load_volume(kind, volume_dir).affine


# Get Modality/Segmentation Volume Header
def get_volume_header(kind, volume_dir):
    return load_volume(kind, volume_dir).header

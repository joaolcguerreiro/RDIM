import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# Save CT Scan Volume
def save_volume(volume, save_dir, file_name, affine, header):
    # Create Save Directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nib.save(
        nib.Nifti1Image(volume, affine, header=header),
        os.path.join(save_dir, file_name + ".nii.gz"),
    )


# Get CT Scan Volume
def get_volume_data(volume_dir):
    return np.clip(nib.load(volume_dir).get_fdata().astype(np.float32)[:, :, :], a_min=0, a_max=None)


# Get CT Scan Volume Affine
def get_volume_affine(volume_dir):
    return nib.load(volume_dir).affine


# Get CT Scan Volume Header
def get_volume_header(volume_dir):
    return nib.load(volume_dir).header


# Get CT Scan Volume Slice (2D)
def get_rgb_volume_slice(slice_index, plane, volume):
    if plane == "axial":
        # 3D volume (Y, X, Z, C) => Slice along z-axis => 2D image (Y, X, C) => For display purposes interchange axis => 2D image (X, Y, C)
        return volume[:, :, slice_index, :].swapaxes(0, 1)  # We get 2d plane (X, Y, C)
    elif plane == "sagittal":
        # 3D volume (Y, X, Z, C) => Slice along y-axis => 2D image (X, Z, C) => For display purposes interchange axis => 2D image (Z, X, C)
        return volume[slice_index, :, :, :].swapaxes(0, 1)  # We get 2d plane (Z, X, C)
    elif plane == "coronal":
        # 3D volume (Y, X, Z, C) => Slice along x-axis => 2D image (Y, Z, C) => For display purposes interchange axis => 2D image (Z, Y, C)
        return volume[:, slice_index, :, :].swapaxes(0, 1)  # We get 2d plane (Z, Y, C)
    else:
        return NotImplementedError('Invalid plane')


# Get Number of Slices in CT Scan Volume Plane
def get_slices_number(plane, volume):
    if plane == "axial":
        return volume.shape[2]
    elif plane == "sagittal":
        return volume.shape[0]
    elif plane == "coronal":
        return volume.shape[1]
    else:
        return NotImplementedError('Invalid plane')


def volume_to_grayscale(volume):
    # (Z, X, Y) => (Y, X, Z) => (Y, X, Z, C) => (Y, X, Z, 1)
    return np.expand_dims(volume.swapaxes(0, -1), axis=-1)


def display_2d(imgs, n_rows=1, n_cols=1, fig_size=(2, 2)):
    _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows), squeeze=False)

    for index in range(n_rows * n_cols):
        ax[index // n_cols, index % n_cols].axis('off')

    for index, img in enumerate(imgs):
        ax[index // n_cols, index % n_cols].imshow(img, cmap='gray')
        ax[index // n_cols, index % n_cols].axis('off')

    plt.tight_layout()
    plt.show()


def display_3d(volumes, n_rows=1, n_cols=1, fig_size=(750, 750)):
    pass


def display_volumes(slice_index, volume_index, plane, volumes, n_rows, n_cols, dimension):
    if dimension == '2D':
        if int(slice_index) >= 0:
            print('Volume {}... Slice {}...'.format(f'{volume_index:05}', slice_index))
            imgs = [get_rgb_volume_slice(slice_index, plane, volume) for volume in volumes]
            display_2d(imgs, n_rows, n_cols)
        else:
            for slice_index in range(get_slices_number(plane, volumes[0])):
                print('Volume {}... Slice {}...'.format(f'{volume_index:05}', slice_index))
                imgs = [get_rgb_volume_slice(slice_index, plane, volume) for volume in volumes]
                display_2d(imgs, n_rows, n_cols)
    elif dimension == '3D':
        print('Volume {}...'.format(f'{volume_index:05}'))
        display_3d(volumes, n_rows, n_cols)
    else:
        raise NotImplementedError('Invalid dimension')


def display_batch(batch, plane, dimension, slice_index=-1, volume_index=-1):
    volumes = list(volume_to_grayscale(volume[0, :, :, :]) for volume in list(volume for volume in batch))

    display_volumes(slice_index, volume_index, plane, volumes, 2, int(len(volumes) / 2), dimension)

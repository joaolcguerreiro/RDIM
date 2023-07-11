import os
import shutil
import logging
import nibabel as nib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import monai.transforms as transforms
from glob import glob
from tqdm import tqdm
from matplotlib.patches import Patch


# Load CT Scan Volume
def load_volume(volume_dir):
    return nib.load(volume_dir)


# Save CT Scan Volume
def save_volume(volume, file_name, affine, header, save_dir):
    # Create Save Directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nib.save(
        nib.Nifti1Image(volume, affine, header=header),
        os.path.join(save_dir, file_name + '.nii.gz'),
    )


# Get CT Scan Volume Data
def get_volume_data(volume_dir):
    return load_volume(volume_dir).get_fdata().astype(np.float32)[:, :, :]


# Get CT Scan Volume Affine
def get_volume_affine(volume_dir):
    return load_volume(volume_dir).affine


# Get CT Scan Volume Header
def get_volume_header(volume_dir):
    return load_volume(volume_dir).header


# Get CT Scan Volume Metadata
def get_metadata(volume_dir):
    return {
        'affine': get_volume_affine(volume_dir),
        'header': get_volume_header(volume_dir)
    }


# Get CT Scan Volume Slice (2D)
def slice_volume(slice_index, plane, volume):
    if plane == 'axial':
        # 3D volume (Y, X, Z, C) => Slice along z-axis => 2D image (Y, X, C) => For display purposes interchange axis => 2D image (X, Y, C)
        return volume[:, :, slice_index, :].swapaxes(0, 1)  # We get 2d plane (X, Y, C)
    elif plane == 'sagittal':
        # 3D volume (Y, X, Z, C) => Slice along y-axis => 2D image (X, Z, C) => For display purposes interchange axis => 2D image (Z, X, C)
        return volume[slice_index, :, :, :].swapaxes(0, 1)  # We get 2d plane (Z, X, C)
    elif plane == 'coronal':
        # 3D volume (Y, X, Z, C) => Slice along x-axis => 2D image (Y, Z, C) => For display purposes interchange axis => 2D image (Z, Y, C)
        return volume[:, slice_index, :, :].swapaxes(0, 1)  # We get 2d plane (Z, Y, C)
    else:
        return NotImplementedError('Invalid plane')


# Get Number of Slices in CT Scan Volume Plane
def count_volume_slices(plane, volume):
    if plane == 'axial':
        return volume.shape[2]
    elif plane == 'sagittal':
        return volume.shape[0]
    elif plane == 'coronal':
        return volume.shape[1]
    else:
        return NotImplementedError('Invalid plane')


def adjust_volume_orientation(volume):
    # (1, Z, X, Y) => (Y, Z, X, 1) => (Y, X, Z, 1)
    return volume.swapaxes(0, -1).swapaxes(1, 2)


def draw_image_2d(imgs, n_rows=1, n_cols=1, fig_size=(3, 3), labels=None, show=True, ax=None, theme='light'):
    if theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows), squeeze=False)

    for index in range(ax.size):
        ax[index // ax.shape[1], index % ax.shape[1]].axis('off')

    for index, img in enumerate(imgs):
        ax[index // ax.shape[1], index % ax.shape[1]].imshow(img, cmap='gray')

        if labels and index < len(labels):
            legend_elements = [Patch(label=labels[index])]
            ax[index // ax.shape[1], index % ax.shape[1]].legend(handles=legend_elements, loc='upper right', handlelength=0, handleheight=0, handletextpad=0)

    plt.tight_layout() if show else None
    plt.show() if show else None


def draw_image_3d(volumes, n_rows=1, n_cols=1, fig_size=(750, 750), labels=None, show=True, ax=None, theme='light'):
    pass


def display_volumes(volumes, plane, dimension, n_rows, n_cols, volume_indexes=None, slice_index=-1, labels=None, theme='light'):
    if dimension == '2D':
        if int(slice_index) >= 0:
            print(f'Volume {", ".join([f"{index:05}" for index in (volume_indexes or ["Unknown"])])}... Slice {slice_index}...')
            imgs = [slice_volume(slice_index, plane, volume) for volume in volumes]
            draw_image_2d(imgs, n_rows, n_cols, labels=labels, theme=theme)
        else:
            for slice_index in range(count_volume_slices(plane, volumes[0])):
                print(f'Volume {", ".join([f"{index:05}" for index in (volume_indexes or ["Unknown"])])}... Slice {slice_index}...')
                imgs = [slice_volume(slice_index, plane, volume) for volume in volumes]
                draw_image_2d(imgs, n_rows, n_cols, labels=labels, theme=theme)
    elif dimension == '3D':
        print(f'Volume {", ".join([f"{index:05}" for index in (volume_indexes or ["Unknown"])])}...')
        draw_image_3d(volumes, n_rows, n_cols, labels=labels, theme=theme)
    else:
        raise NotImplementedError('Invalid dimension')


def compute_data_statistics(data_dir):
    print("=" * 115)

    _min = float('+inf')
    _max = float('-inf')
    mean = np.zeros(1)
    std = np.zeros(1)

    # Fetch all data file paths
    dataset_files = sorted(glob(os.path.join(data_dir, '*')))

    # Define progress bar
    pbar = tqdm(dataset_files, desc="Computing Data Statistics")

    for index, file in enumerate(pbar):
        # Load file
        volume = get_volume_data(file)

        _min = min(_min, np.min(volume))
        _max = max(_max, np.max(volume))
        mean += np.mean(volume)
        std += np.std(volume)

        if index == len(pbar) - 1:
            mean = mean.item() / len(dataset_files)
            std = std.item() / len(dataset_files)
            pbar.set_postfix({"Min": f"{_min:.4f}", "Max": f"{_max:.4f}", "Mean": f"{mean:.4f}", "STD": f"{std:.4f}"})

    # Logging the progress bar
    logging.info(pbar)

    return _min, _max, mean, std


# Compute Super-Resolved CT Scan Volume Metadata
def compute_super_resolution_metadata(lr_metadata, scale_factor):
    # Create a copy of the header
    super_resolution_header = lr_metadata['header'].copy()
    super_resolution_header.set_zooms(np.array(super_resolution_header.get_zooms()) / scale_factor)
    super_resolution_header.set_data_shape(np.array(super_resolution_header.get_data_shape()) * scale_factor)

    # Update the affine matrix to account for the new voxel dimensions
    super_resolution_affine = lr_metadata['affine'].copy()
    np.fill_diagonal(super_resolution_affine[:3, :3], np.diagonal(super_resolution_affine[:3, :3]) / scale_factor)

    return {
        'affine': super_resolution_affine,
        'header': super_resolution_header
    }


def compute_batch_super_resolution_metadata(lr_metadata, scale_factor):
    batch_sr_metadata = {
        'affine': [],
        'header': []
    }

    for affine, header in zip(lr_metadata['affine'], lr_metadata['header']):
        sr_metadata = compute_super_resolution_metadata({'affine': affine, 'header': header}, scale_factor)

        batch_sr_metadata['affine'].append(sr_metadata['affine'])
        batch_sr_metadata['header'].append(sr_metadata['header'])

    return batch_sr_metadata


def denormalize(volume, mean, std):
    return transforms.NormalizeIntensity(subtrahend=-mean/std, divisor=1/std)(volume)


def upsample(volume, scale_factor):
    return transforms.Zoom(zoom=scale_factor, mode="trilinear", keep_size=False)(volume)


def compute_scale_factor(lr_volume, gt_volume):
    # Remove channel dimension and fetch volume shapes
    lr_shape = lr_volume.squeeze(axis=0).shape
    gt_shape = gt_volume.squeeze(axis=0).shape

    # Calculate the scale factor for each dimension
    scale_factor = np.array(gt_shape) / np.array(lr_shape)

    return scale_factor


def display_batch(lr_volumes=None, sr_volumes=None, gt_volumes=None, plane='axial', dimension='2D', volume_indexes=None, slice_index=-1, labels=True, theme='light'):
    lr_volumes = list() if lr_volumes is None else list(lr_volumes)
    sr_volumes = list() if sr_volumes is None else list(sr_volumes)
    gt_volumes = list() if gt_volumes is None else list(gt_volumes)

    if not all(len(_list) == 0 or len(_list) == max(len(lr_volumes), len(gt_volumes), len(sr_volumes)) for _list in [lr_volumes, gt_volumes, sr_volumes]):
        raise ValueError("The numbers of low-resolution, super-resolution and ground truth volumes do not match.")

    if gt_volumes and sr_volumes and not (gt_volumes[0].shape == sr_volumes[0].shape):
        raise ValueError("The shapes of the super-resolution and ground truth volumes do not match.")

    # Compute scale factor
    hr_volumes = gt_volumes if gt_volumes else sr_volumes
    scale_factor = compute_scale_factor(lr_volumes[0], hr_volumes[0]) if (lr_volumes and hr_volumes) else (1, 1, 1)

    # Reorient volume for visualization
    lr_volumes = list(adjust_volume_orientation(upsample(lr_volume, scale_factor)) for lr_volume in lr_volumes)
    sr_volumes = list(adjust_volume_orientation(sr_volume) for sr_volume in sr_volumes)
    gt_volumes = list(adjust_volume_orientation(gt_volume) for gt_volume in gt_volumes)

    # Number of series and samples per series
    num_series = sum(1 for _array in [lr_volumes, sr_volumes, gt_volumes] if _array)
    batch_size = len(lr_volumes or sr_volumes or gt_volumes)
    _labels = list(label for label, _array in zip(['Downsampled', 'Synthetic', 'Ground Truth'], [lr_volumes, sr_volumes, gt_volumes]) if _array for _ in range(batch_size)) if labels else None

    display_volumes(lr_volumes + sr_volumes + gt_volumes, plane=plane, dimension=dimension, n_rows=num_series, n_cols=batch_size, volume_indexes=volume_indexes, slice_index=slice_index, labels=_labels, theme=theme)


def save_batch(batch, batch_metadata, batch_index, save_dir_path, mean=None, std=None):
    for volume_index, (volume, affine, header) in enumerate(zip(batch, batch_metadata['affine'], batch_metadata['header'])):
        # Denormalize volume
        volume = volume if (mean is None or std is None) else denormalize(volume, mean=mean, std=std)

        save_volume(volume, affine, header, str(volume_index * batch_index), save_dir_path)


def save_sample(gen_volume, gt_volume, save_dir_path, mean=None, std=None, plane='axial', fig_size=(6, 6), theme='light'):
    # Reorient volume for visualization
    gen_volume = adjust_volume_orientation(gen_volume)
    gt_volume = adjust_volume_orientation(gt_volume)

    # Denormalize volumes
    gen_volume = gen_volume if (mean is None or std is None) else denormalize(gen_volume, mean=mean, std=std)
    gt_volume = gt_volume if (mean is None or std is None) else denormalize(gt_volume, mean=mean, std=std)

    # Define slice index
    slice_index = count_volume_slices(plane=plane, volume=gt_volume) // 2

    gen_slice = slice_volume(slice_index=slice_index, plane=plane, volume=gen_volume)
    gt_slice = slice_volume(slice_index=slice_index, plane=plane, volume=gt_volume)

    # Create a new figure
    fig = plt.figure(figsize=fig_size)

    if theme == 'dark':
        plt.style.use('dark_background')

    # Stick images always to top corners despite figure ratio
    n_imgs = 2
    grid_img_top = 1
    grid_img_bottom = grid_img_top - (fig_size[0] / fig_size[1] / n_imgs)

    # Create grid for images
    grid_img = plt.GridSpec(1, n_imgs, wspace=0, left=0, right=1, bottom=grid_img_bottom, top=grid_img_top)
    ax_img = np.array([[fig.add_subplot(grid_img[0]), fig.add_subplot(grid_img[1])]])

    # Create grid for histogram
    grid_hist = plt.GridSpec(1, n_imgs, top=grid_img_bottom * 0.875)
    ax_hist = fig.add_subplot(grid_hist[0:])

    # Draw slices
    draw_image_2d([gen_slice, gt_slice], labels=['Synthetic', 'Ground Truth'], show=False, ax=ax_img, theme=theme)

    # Draw pixel distribution histogram
    draw_histogram(
        data=[
            {'Pixel Intensity': gen_slice.flatten().tolist(), 'Data Type': ['Synthetic'] * gen_slice.numel()},
            {'Pixel Intensity': gt_slice.flatten().tolist(), 'Data Type': ['Ground Truth'] * gt_slice.numel()},
        ],
        title='Sample Pixel Distribution',
        x_label='Pixel Intensity',
        hue='Data Type',
        element='step',
        fill=False,
        ax=ax_hist,
        theme=theme
    )

    # Save figure
    plt.savefig(save_dir_path)


def draw_line_chart(data, title, x_label, y_label, save_dir_path, ax=None, theme='light'):
    if theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots()

    for _dict in data:
        ax.plot(_dict[x_label], _dict[y_label], label=_dict['Label'])

    # Chart customization
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    # Save chart
    if save_dir_path:
        plt.savefig(save_dir_path)


def draw_histogram(data, title, x_label, hue=None, stat='count', element='bars', fill=True, save_dir_path=None, ax=None, theme='light'):
    # Parse data
    data = pd.DataFrame({
        key: sum((_dict[key] for _dict in data), []) for key in [x_label, hue] if key is not None
    })

    if theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots()

    # Chart customization
    sns.histplot(data=data, x=x_label, hue=hue, stat=stat, element=element, fill=fill, palette="Pastel1")
    ax.set_title(title)

    # Save chart
    if save_dir_path:
        plt.savefig(save_dir_path)


def draw_box_plot(data, title, x_label, y_label, hue=None, save_dir_path=None, ax=None, theme='light'):
    # Parse data
    data = pd.DataFrame({
        key: sum((_dict[key] for _dict in data), []) for key in [x_label, y_label, hue] if key is not None
    })

    if theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots()

    # Chart customization
    sns.boxplot(data=data, x=x_label, y=y_label, hue=hue, palette="Pastel1")
    ax.set_title(title)

    # Save chart
    if save_dir_path:
        plt.savefig(save_dir_path)


def create_experiment_directory(experiment_path, config_path):
    # Create experiment folder
    os.makedirs(experiment_path)

    # Create checkpoints, figures, logs and results folder
    os.makedirs(os.path.join(experiment_path, 'checkpoints'))
    os.makedirs(os.path.join(experiment_path, 'results'))

    for path in [os.path.join('figures', 'train'), os.path.join('figures', 'test')]:
        os.makedirs(os.path.join(experiment_path, path))

    for path in [os.path.join('logs', 'train'), os.path.join('logs', 'test')]:
        os.makedirs(os.path.join(experiment_path, path))

    # Clone global config file for reproducibility
    shutil.copy(config_path, experiment_path)

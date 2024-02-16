import os
import sys
import shutil
import logging
import torch
import random
import nibabel as nib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import monai.transforms as transforms
from glob import glob
from tqdm import tqdm
from matplotlib.patches import Patch
sys.path.append(os.path.join('..', '..'))
from src.utils.logger import print_divisor
from src.utils.transforms import NormalizeIntensity, ThresholdIntensity, EnsureExactChannels


def load_volume(volume_dir):
    return nib.load(volume_dir)


def save_volume(volume, affine, header, filename, save_dir):
    # Create Save Directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nib.save(
        nib.Nifti1Image(volume, affine=affine, header=header),
        os.path.join(save_dir, filename),
    )


def get_volume_data(volume_dir):
    return load_volume(volume_dir).get_fdata().astype(np.float32)[:, :, :]


def get_volume_affine(volume_dir):
    return load_volume(volume_dir).affine


def get_volume_header(volume_dir):
    return load_volume(volume_dir).header


def get_volume_metadata(volume_dir):
    return {
        'filename': os.path.basename(volume_dir),
        'affine': get_volume_affine(volume_dir),
        'header': get_volume_header(volume_dir)
    }


def slice_volume(slice_index, plane, volume):
    """
    Get Volume Slice (2D).
    Expected Volume Dimensions Order is (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) = (Z, Y, X, C).

    Note:
        X-axis: Extends in (Left - Right) direction, and it is perpendicular to Sagittal plane.
        Y-axis: Extends in (Anterior - Posterior) direction, and it is perpendicular to Coronal plane.
        Z-axis: Extends in (Inferior - Superior) direction, and it is perpendicular to Axial plane.
    """

    if plane == 'axial':
        # 3D volume (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) => Slice along Longitudinal Axis => 2D image (Anteroposterior Axis, Mediolateral Axis, Channel) = (Y, X, C)
        return volume[slice_index, :, :, :]     # For display purposes keep the orientation
    elif plane == 'coronal':
        # 3D volume (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) => Slice along Anteroposterior Axis => 2D image (Longitudinal Axis, Mediolateral Axis, Channel) = (Z, X, C)
        return volume[:, slice_index, :, :]     # For display purposes keep the orientation
    elif plane == 'sagittal':
        # 3D volume (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) => Slice along Mediolateral Axis => 2D image (Longitudinal Axis, Anteroposterior Axis, Channel) = (Z, Y, C)
        return volume[:, :, slice_index, :]     # For display purposes keep the orientation
    else:
        raise NotImplementedError('Invalid plane')


def get_num_plane_slices(plane, volume):
    """
    Get Number of Slices in Volume Plane.
    Expected Volume Dimensions Order is (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) = (Z, Y, X, C).

    Note:
        X-axis: Extends in (Left - Right) direction, and it is perpendicular to Sagittal plane.
        Y-axis: Extends in (Anterior - Posterior) direction, and it is perpendicular to Coronal plane.
        Z-axis: Extends in (Inferior - Superior) direction, and it is perpendicular to Axial plane.
    """

    if plane == 'axial':
        return volume.shape[0]
    elif plane == 'coronal':
        return volume.shape[1]
    elif plane == 'sagittal':
        return volume.shape[2]
    else:
        raise NotImplementedError('Invalid plane')


def reorder_volume_dimensions(volume, channel_dim=0, longitudinal_axis_dim=1, anteroposterior_axis_dim=2, mediolateral_axis_dim=3):
    """
    Rearrange Volume Dimensions Order from an Arbitrary Order of Dimensions into (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) = (Z, Y, X, C).
    Expected Volume Dimensions Order is Arbitrary.

    Note:
        X-axis: Extends in (Left - Right) direction, and it is perpendicular to Sagittal plane.
        Y-axis: Extends in (Anterior - Posterior) direction, and it is perpendicular to Coronal plane.
        Z-axis: Extends in (Inferior - Superior) direction, and it is perpendicular to Axial plane.
    """

    # Reorder arbitrary volume dimensions into (Longitudinal Axis, Anteroposterior Axis, Mediolateral Axis, Channel) = (Z, Y, X, C)
    return volume.permute(longitudinal_axis_dim, anteroposterior_axis_dim, mediolateral_axis_dim, channel_dim).contiguous()


def format_volume(volume, _min=None, _max=None):
    """
    Adjust Volume for Visualization.
    Expected Volume Dimensions Order is Flexible, but Channel Dimension is required to be First (Channel, ...) = (C, ...).
    """

    # Scale volume to the valid range for imshow with RGB data ([0, 1] for floats or [0, 255] for integers).
    volume = (min_max_scaling(volume, _min=_min, _max=_max) if (_min is not None and _max is not None) else ((volume - volume.min()) / (volume.max() - volume.min()))) * (255 if not torch.is_floating_point(volume) else 1.0)

    # Ensure valid number of channels (Grayscale: 1, RGB: 3, RGBA: 4)
    if volume.shape[0] not in (1, 3, 4):
        volume = force_valid_channels(volume, num_channel=1)

    # Reorder volume dimensions for visualization
    volume = reorder_volume_dimensions(volume)

    return volume


def draw_image_2d(imgs, n_rows=1, n_cols=1, labels=None, show=True, ax=None, fig_size=(3, 3), theme='light'):
    if theme == 'light':
        plt.style.use('default')
    elif theme == 'dark':
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


def draw_image_3d(volumes, n_rows=1, n_cols=1, labels=None, show=True, ax=None, fig_size=(3, 3), theme='light'):
    pass


def display_volumes(volumes, plane, dimension, n_rows=None, n_cols=None, volume_filenames=None, slice_index=-1, labels=None, show=True, ax=None, fig_size=(3, 3), theme='light'):
    if dimension == '2D':
        if int(slice_index) >= 0:
            slice_indexes = [slice_index]
        else:
            slice_indexes = range(get_num_plane_slices(plane, volumes[0]))

        for slice_index in slice_indexes:
            print(f'Volumes at Slice {slice_index}:\n' + '\n'.join(f'- {volume_filename}' for volume_filename in volume_filenames)) if show else None
            imgs = [slice_volume(slice_index, plane, volume) for volume in volumes]
            draw_image_2d(imgs, n_rows, n_cols, labels=labels, show=show, ax=ax, fig_size=fig_size, theme=theme)
    elif dimension == '3D':
        print('Volumes:\n' + '\n'.join(f'- {volume_filename}' for volume_filename in volume_filenames)) if show else None
        draw_image_3d(volumes, n_rows, n_cols, labels=labels, show=show, ax=ax, fig_size=fig_size, theme=theme)
    else:
        raise NotImplementedError('Invalid dimension')


def compute_data_statistics(data_dir, num_channels):
    print_divisor()

    # Fetch all data file paths
    dataset_files = sorted(glob(os.path.join(data_dir, '*')))

    _min = np.full(num_channels, float('+inf'))
    _max = np.full(num_channels, float('-inf'))
    mean = np.zeros(num_channels)
    std = np.zeros(num_channels)

    # Define progress bar
    pbar = tqdm(dataset_files, desc="Computing Data Statistics")

    for index, file in enumerate(pbar):
        # Load file
        volume = get_volume_data(file)

        # Ensure multichannel data
        if volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)

        for channel in range(num_channels):
            _min[channel] = min(_min[channel], np.min(volume[channel]))
            _max[channel] = max(_max[channel], np.max(volume[channel]))
            mean[channel] += np.mean(volume[channel])
            std[channel] += np.std(volume[channel])

        if index == len(pbar) - 1:
            mean /= len(dataset_files)
            std /= len(dataset_files)
            pbar.set_postfix({
                "Min": str(np.squeeze(np.around(_min, 2)).tolist()),
                "Max": str(np.squeeze(np.around(_max, 2)).tolist()),
                "Mean": str(np.squeeze(np.around(mean, 2)).tolist()),
                "STD": str(np.squeeze(np.around(std, 2)).tolist())
            })

    # Logging the progress bar
    logging.info(pbar)
    print_divisor(last=True)

    return _min, _max, mean, std


def compute_super_resolution_volume_metadata(lr_metadata, scale_factor=None):
    super_resolution_header = None
    super_resolution_affine = None

    if scale_factor is not None and scale_factor > 0:
        # Create copies of the header and affine matrix
        super_resolution_header = lr_metadata['header'].copy()
        super_resolution_affine = lr_metadata['affine'].copy()

        # Update the header to account for the new voxel dimensions
        super_resolution_header.set_zooms(np.array(super_resolution_header.get_zooms()) / scale_factor)
        super_resolution_header.set_data_shape(np.array(super_resolution_header.get_data_shape()) * scale_factor)

        # Update the affine matrix to account for the new voxel dimensions
        np.fill_diagonal(super_resolution_affine[:3, :3], np.diagonal(super_resolution_affine[:3, :3]) / scale_factor)

    return {
        'filename': lr_metadata['filename'],
        'affine': super_resolution_affine,
        'header': super_resolution_header
    }


def min_max_scaling(volume, _min, _max):
    return NormalizeIntensity(mean=_min, std=(_max-_min), channel_wise=True)(volume)


def denormalize(volume, mean, std):
    return NormalizeIntensity(mean=-mean/std, std=1/std, channel_wise=True)(volume)


def upsample(volume, scale_factor):
    return transforms.Zoom(zoom=scale_factor, mode="trilinear", keep_size=False)(volume)


def clamp(volume, _min, _max, mean=None, std=None):
    _min = _min if (mean is None or std is None) else ((_min - mean) / std)
    _max = _max if (mean is None or std is None) else ((_max - mean) / std)

    return ThresholdIntensity(threshold=_max, above=False, channel_wise=True, cval=_max)(ThresholdIntensity(threshold=_min, above=True, channel_wise=True, cval=_min)(volume))


def force_valid_channels(volume, num_channel):
    return EnsureExactChannels(num_channels=num_channel)(volume)


def compute_scale_factor(lr_volume, hr_volume):
    # Fetch volume spatial dimensions (discard channel dimension)
    lr_shape = lr_volume.shape[1:]
    hr_shape = hr_volume.shape[1:]

    # Calculate the scale factor for each dimension
    scale_factor = np.array(hr_shape) / np.array(lr_shape)

    return scale_factor


def display_batch(lr_volumes=None, sr_volumes=None, gt_volumes=None, _min=None, _max=None, plane='axial', dimension='2D', volume_filenames=None, slice_index=-1, labels=True, fig_size=(3, 3), theme='light'):
    lr_volumes = list() if lr_volumes is None else list(lr_volumes.to('cpu'))
    sr_volumes = list() if sr_volumes is None else list(sr_volumes.to('cpu'))
    gt_volumes = list() if gt_volumes is None else list(gt_volumes.to('cpu'))

    if not all(len(_list) == 0 or len(_list) == max(len(lr_volumes), len(gt_volumes), len(sr_volumes)) for _list in [lr_volumes, gt_volumes, sr_volumes]):
        raise ValueError("The numbers of low-resolution, super-resolution and ground truth volumes do not match.")

    if gt_volumes and sr_volumes and not (gt_volumes[0].shape == sr_volumes[0].shape):
        raise ValueError("The shapes of the super-resolution and ground truth volumes do not match.")

    # Compute scale factor
    hr_volumes = gt_volumes if gt_volumes else sr_volumes
    scale_factor = compute_scale_factor(lr_volumes[0], hr_volumes[0]) if (lr_volumes and hr_volumes) else (1, 1, 1)

    # Adjust volumes for visualization
    lr_volumes = list(format_volume(upsample(lr_volume, scale_factor), _min=_min, _max=_max) for lr_volume in lr_volumes)
    sr_volumes = list(format_volume(sr_volume, _min=_min, _max=_max) for sr_volume in sr_volumes)
    gt_volumes = list(format_volume(gt_volume, _min=_min, _max=_max) for gt_volume in gt_volumes)

    # Interleave volumes for visualization
    volumes = [volume for interleaved_volumes in zip(*list(filter(None, [lr_volumes, sr_volumes, gt_volumes]))) for volume in interleaved_volumes]

    # Number of series (volume types) and samples per series (batch size)
    num_volume_types = sum(1 for _array in [lr_volumes, sr_volumes, gt_volumes] if _array)
    batch_size = len(lr_volumes or sr_volumes or gt_volumes)
    _labels = list(label for label, _array in zip(['Downsampled', 'Synthetic', 'Ground Truth'], [lr_volumes, sr_volumes, gt_volumes]) if _array) * batch_size if labels else None

    print_divisor(width=50)

    display_volumes(volumes, plane=plane, dimension=dimension, n_rows=batch_size, n_cols=num_volume_types, volume_filenames=volume_filenames, slice_index=slice_index, labels=_labels, fig_size=fig_size, theme=theme)

    print_divisor(width=50)


def save_batch(volumes, volumes_metadata, save_dir_path, mean=None, std=None):
    for volume, filename, affine, header in zip(volumes, volumes_metadata['filename'], volumes_metadata['affine'], volumes_metadata['header']):
        # Denormalize volume
        volume = volume if (mean is None or std is None) else denormalize(volume, mean=mean, std=std)

        save_volume(volume.to('cpu'), affine, header, filename, save_dir_path)


def collect_batch_pixels(volumes, mean=None, std=None):
    volumes_pixels = []

    for volume in volumes:
        # Denormalize volume
        volume = volume if (mean is None or std is None) else denormalize(volume, mean=mean, std=std)

        volumes_pixels.extend(volume.to('cpu').flatten().tolist())

    return volumes_pixels


def parse_bin_range(_min, _max):
    return float(torch.tensor(_min).min()), float(torch.tensor(_max).max())


def compute_max_frequency(pixel_values, bins, _min=None, _max=None):
    return torch.histc(torch.tensor(pixel_values), bins=bins, min=float(torch.tensor(_min if _min is not None else 0).min()), max=float(torch.tensor(_max if _max is not None else 0).max())).max().item() if (torch.tensor(pixel_values).numel() > 0 and isinstance(bins, int)) else None


def save_sample(save_filename, lr_volume, gen_volume, gt_volume=None, bins='auto', _min=None, _max=None, mean=None, std=None, plane='axial', dimension='2D', fig_size=(3, 3), theme='light', with_histogram=True, bound_range=True):
    # Denormalize volumes
    lr_volume = lr_volume if (mean is None or std is None) else denormalize(lr_volume, mean=mean, std=std)
    gen_volume = gen_volume if (mean is None or std is None) else denormalize(gen_volume, mean=mean, std=std)
    gt_volume = gt_volume if (mean is None or std is None or gt_volume is None) else denormalize(gt_volume, mean=mean, std=std)

    # Compute scale factor
    hr_volume = gen_volume if gen_volume is not None else gt_volume
    scale_factor = compute_scale_factor(lr_volume, hr_volume) if (lr_volume is not None and hr_volume is not None) else (1, 1, 1)

    # Adjust volumes for visualization
    lr_volume = format_volume(raw_lr_volume := upsample(lr_volume, scale_factor).to('cpu'), _min=_min, _max=_max)
    gen_volume = format_volume(raw_gen_volume := gen_volume.to('cpu'), _min=_min, _max=_max)
    gt_volume = format_volume(raw_gt_volume := gt_volume.to('cpu'), _min=_min, _max=_max) if gt_volume is not None else (raw_gt_volume := gt_volume)

    # Define slice index
    slice_index = (get_num_plane_slices(plane=plane, volume=gen_volume) // 2) if dimension == '2D' else None

    # Create a new figure
    n_columns = sum(1 for volume in [lr_volume, gen_volume, gt_volume] if volume is not None)
    n_img_rows = 1
    n_hist_rows = (2 if with_histogram else 0)
    fig_width = fig_size[0] * n_columns
    fig_height = fig_size[1] * (n_img_rows + n_hist_rows)

    fig = plt.figure(figsize=(fig_width, fig_height))

    if theme == 'light':
        plt.style.use('default')
    elif theme == 'dark':
        plt.style.use('dark_background')

    # Define images grid position
    grid_img_top = 1
    grid_img_bottom = grid_img_top - (n_img_rows / (n_img_rows + n_hist_rows))

    # Create grid for images
    grid_img = plt.GridSpec(n_img_rows, n_columns, wspace=0, hspace=0, left=0, right=1, bottom=grid_img_bottom, top=grid_img_top)
    ax_img = np.array([[fig.add_subplot(grid_img[row, column]) for row in range(n_img_rows) for column in range(n_columns)]])

    # Draw slices
    volumes, labels = zip(*filter(lambda slice_label_pair: slice_label_pair[0] is not None, zip([lr_volume, gen_volume, gt_volume], ['Downsampled', 'Synthetic', 'Ground Truth'])))
    display_volumes(volumes, plane=plane, dimension=dimension, slice_index=slice_index, labels=labels, show=False, ax=ax_img, theme=theme)

    if with_histogram:
        lr_pixel_values = (slice_volume(slice_index=slice_index, plane=plane, volume=reorder_volume_dimensions(raw_lr_volume)) if dimension == '2D' else raw_lr_volume).flatten().tolist()
        gen_pixel_values = (slice_volume(slice_index=slice_index, plane=plane, volume=reorder_volume_dimensions(raw_gen_volume)) if dimension == '2D' else raw_gen_volume).flatten().tolist()
        gt_pixel_values = (slice_volume(slice_index=slice_index, plane=plane, volume=reorder_volume_dimensions(raw_gt_volume)) if dimension == '2D' else raw_gt_volume).flatten().tolist() if raw_gt_volume is not None else list()

        # Create grid for histogram
        grid_hist = plt.GridSpec(n_hist_rows, n_columns, top=grid_img_bottom * 0.875)
        ax_hist = fig.add_subplot(grid_hist[:])

        # Draw pixel distribution histogram
        draw_histogram(
            data=[
                {'Pixel Intensity': lr_pixel_values, 'Data Type': ['Downsampled'] * len(lr_pixel_values)},
                {'Pixel Intensity': gen_pixel_values, 'Data Type': ['Synthetic'] * len(gen_pixel_values)},
                *({'Pixel Intensity': gt_pixel_values, 'Data Type': ['Ground Truth'] * len(gt_pixel_values)} for gt_pixel_values in [gt_pixel_values] if gt_pixel_values),
            ],
            title='Sample Pixel Distribution',
            x_label='Pixel Intensity',
            hue='Data Type',
            bins=bins,
            bin_range=None if (_min is None or _max is None) else parse_bin_range(_min=_min, _max=_max),
            frequency_range=(None, (compute_max_frequency(gt_pixel_values, bins=bins, _min=_min, _max=_max) * 1.15) if (_min is not None and _max is not None) else None) if bound_range else None,
            element='step',
            fill=False,
            ax=ax_hist,
            theme=theme
        )

    # Save figure
    ensure_dir_exists(save_filename)
    plt.savefig(save_filename)


def save_samples(save_dir_path, volumes_metadata, lr_volumes, gen_volumes, gt_volumes=None, bins='auto', _min=None, _max=None, mean=None, std=None, plane='axial', dimension='2D', fig_size=(3, 3), theme='light', with_histogram=False, bound_range=False):
    save_filenames = [os.path.join(save_dir_path, f'{os.path.splitext(filename)[0]}') for filename in volumes_metadata['filename']]

    for lr_volume, gen_volume, gt_volume, save_filename in zip(lr_volumes, gen_volumes, (gt_volumes if gt_volumes is not None else [gt_volumes] * len(gen_volumes)), save_filenames):
        save_sample(
            lr_volume=lr_volume,
            gen_volume=gen_volume,
            gt_volume=gt_volume,
            bins=bins,
            _min=_min, _max=_max, mean=mean, std=std,
            save_filename=save_filename,
            plane=plane,
            dimension=dimension,
            fig_size=fig_size,
            theme=theme,
            with_histogram=with_histogram,
            bound_range=bound_range
        )


def draw_line_chart(data, title, x_label, y_label, save_filename=None, ax=None, fig_size=None, theme='light'):
    if theme == 'light':
        plt.style.use('default')
    elif theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    # Define color palette
    colors = sns.color_palette('tab10', n_colors=len(data))

    for _dict, color in zip(data, colors):
        ax.plot(_dict[x_label], _dict[y_label], label=_dict['Label'], color=color)

    # Chart customization
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    # Save chart
    if save_filename:
        ensure_dir_exists(save_filename)
        plt.savefig(save_filename)


def draw_histogram(data, title, x_label, hue=None, stat='count', bins='auto', bin_range=None, frequency_range=None, element='bars', fill=True, save_filename=None, ax=None, fig_size=None, theme='light'):
    # Parse data
    data = pd.DataFrame({
        key: sum((_dict[key] for _dict in data), []) for key in [x_label, hue] if key is not None
    })

    if theme == 'light':
        plt.style.use('default')
    elif theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    # Chart customization
    sns.histplot(data=data, x=x_label, hue=hue, stat=stat, bins=bins, binrange=bin_range, element=element, fill=fill, palette='tab10')
    ax.set_title(title)
    ax.set_ylim(frequency_range)

    # Save chart
    if save_filename:
        ensure_dir_exists(save_filename)
        plt.savefig(save_filename)


def draw_box_plot(data, title, x_label, y_label, hue=None, save_filename=None, ax=None, fig_size=None, theme='light'):
    # Parse data
    data = pd.DataFrame({
        key: sum((_dict[key] for _dict in data), []) for key in [x_label, y_label, hue] if key is not None
    })

    if theme == 'light':
        plt.style.use('default')
    elif theme == 'dark':
        plt.style.use('dark_background')

    # Create a new figure
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    # Chart customization
    sns.boxplot(data=data, x=x_label, y=y_label, hue=hue, palette='tab10')
    ax.set_title(title)

    # Save chart
    if save_filename:
        ensure_dir_exists(save_filename)
        plt.savefig(save_filename)


def create_experiment_directory(experiment_path, config_path=None, checkpoint_path=None, inference_mode=False):
    # Create experiment folder
    os.makedirs(experiment_path)

    # Create checkpoints, figures, logs and results folder
    os.makedirs(os.path.join(experiment_path, 'checkpoints'))
    os.makedirs(os.path.join(experiment_path, 'results'))

    if not inference_mode:
        for path in [os.path.join('figures', 'train'), os.path.join('figures', 'test')]:
            os.makedirs(os.path.join(experiment_path, path))

        for path in [os.path.join('logs', 'train'), os.path.join('logs', 'test')]:
            os.makedirs(os.path.join(experiment_path, path))

        # Clone global config file for reproducibility
        os.makedirs(os.path.join(experiment_path, 'configs'))
        shutil.copy(config_path, os.path.join(experiment_path, 'configs'))
    else:
        os.makedirs(os.path.join(experiment_path, 'logs'))
        os.makedirs(os.path.join(experiment_path, 'figures'))

        # Clone checkpoint file for reproducibility
        shutil.copy(checkpoint_path, os.path.join(experiment_path, 'checkpoints'))


def set_seed(seed=20):
    torch.manual_seed(seed)
    torch.set_deterministic_debug_mode("default")
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir_exists(file_path):
    dir_path = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

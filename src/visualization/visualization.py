import os
import sys
import ast
import argparse
import torch
from torch.utils.data import DataLoader
from monai.inferers.inferer import SlidingWindowInferer
sys.path.append(os.path.join('..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator
from src.utils.utils import display_batch, set_seed


# Available display modes
modes = {
    'gt_scans': 'Display ground-truth scan counterpart ',
    'lr_scans': 'Display low-resolution scan counterpart',
    'sr_scans': 'Display super-resolution scan counterpart',
    'lr_gt_scans': 'Display low-resolution and ground truth scans jointly',
    'compare_scans': 'Display low-resolution, super-resolution and ground truth scans jointly',
}

# Available anatomical planes
planes = [
    "axial",
    "sagittal",
    "coronal"
]

# Available Display Dimensions
dimensions = [
    '2D',
    '3D'
]


def main():
    # Set seed for reproducibility
    set_seed(20)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if not (args.checkpoint or args.mode in ('gt_scans', 'lr_scans', 'lr_gt_scans')):
        raise NotImplementedError(f'Provide model checkpoint file to visualize with mode \'{args.mode}\'.')

    if not args.lr_dir and args.mode in ('lr_scans', 'sr_scans', 'lr_gt_scans', 'compare_scans'):
        raise NotImplementedError(f'Low-resolution data directory path is required to visualize with mode \'{args.mode}\'.')

    if not args.hr_dir and args.mode in ('gt_scans', 'lr_gt_scans', 'compare_scans'):
        raise NotImplementedError(f'High-resolution data directory path is required to visualize with mode \'{args.mode}\'.')

    if not args.checkpoint:
        # Define Dataset
        dataset = SuperResolutionDataset(lr_data_dir=args.lr_dir, hr_data_dir=args.hr_dir, _min=args.pixel_range[0], _max=args.pixel_range[1], augmentation=args.augmentation)
    else:
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint)

        # Fetch training data statistics
        _min, _max, mean, std = checkpoint['training_data_min'], checkpoint['training_data_max'], checkpoint['training_data_mean'], checkpoint['training_data_std']

        # Define Dataset
        dataset = SuperResolutionDataset(lr_data_dir=args.lr_dir, hr_data_dir=args.hr_dir, _min=(args.pixel_range[0] or _min), _max=(args.pixel_range[1] or _max), mean=mean, std=std, augmentation=args.augmentation)

        # Define model
        generator = Generator(
            in_channels=checkpoint['generator_in_channels'],
            out_channels=checkpoint['generator_out_channels'],
            num_features=checkpoint['generator_num_features'],
            num_growth_features=checkpoint['generator_num_growth_features'],
            num_rrdb_blocks=checkpoint['generator_num_rrdb_blocks'],
            num_rdb_blocks=checkpoint['generator_num_rdb_blocks'],
            num_rdb_conv_blocks=checkpoint['generator_num_rdb_conv_blocks'],
            scale_factor=checkpoint['generator_scale_factor'],
        ).to(device)

        # Define patch-based inferer
        generator_inferer = SlidingWindowInferer(roi_size=checkpoint['training_lr_patch_size'], overlap=0.25, mode='gaussian', cval=_min, sw_device=device, device=device)

        # Initialize generator weights
        generator.load_state_dict(checkpoint['generator_state_dict'])

        # Set the model to evaluation mode
        generator.eval()

    # Define Dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)

    # Generate list of consecutive indexes for a given batch
    volume_indexes = lambda _batch_index, batch_size: [index for index in range(_batch_index * batch_size, (_batch_index + 1) * batch_size)]

    with torch.no_grad():
        if args.mode == 'gt_scans':
            if int(args.volume) >= 0:
                display_batch(gt_volumes=[dataset[args.volume][1]], plane=args.plane, dimension=args.dim, volume_indexes=[args.volume], slice_index=args.slice, labels=args.labels)

            else:
                for batch_index, (_, hr_volumes, _) in enumerate(dataloader):
                    display_batch(gt_volumes=hr_volumes, plane=args.plane, dimension=args.dim, volume_indexes=volume_indexes(batch_index, args.batch_size), slice_index=args.slice, labels=args.labels)

        elif args.mode == 'lr_scans':
            if int(args.volume) >= 0:
                display_batch(lr_volumes=[dataset[args.volume][0]], plane=args.plane, dimension=args.dim, volume_indexes=[args.volume], slice_index=args.slice, labels=args.labels)

            else:
                for batch_index, (lr_volumes, _, _) in enumerate(dataloader):
                    display_batch(lr_volumes=lr_volumes, plane=args.plane, dimension=args.dim, volume_indexes=volume_indexes(batch_index, args.batch_size), slice_index=args.slice, labels=args.labels)

        elif args.mode == 'sr_scans':
            if int(args.volume) >= 0:
                # Add batch dimension and move inputs to the designated device
                lr_volume = dataset[args.volume][0].unsqueeze(0).to(device)

                display_batch(sr_volumes=generator_inferer(inputs=lr_volume, network=generator), plane=args.plane, dimension=args.dim, volume_indexes=[args.volume], slice_index=args.slice, labels=args.labels)

            else:
                for batch_index, (lr_volumes, _, _) in enumerate(dataloader):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)

                    display_batch(sr_volumes=generator_inferer(inputs=lr_volumes, network=generator), plane=args.plane, dimension=args.dim, volume_indexes=volume_indexes(batch_index, args.batch_size), slice_index=args.slice, labels=args.labels)

        elif args.mode == 'lr_gt_scans':
            if int(args.volume) >= 0:
                display_batch(lr_volumes=[dataset[args.volume][0]], gt_volumes=[dataset[args.volume][1]], plane=args.plane, dimension=args.dim, volume_indexes=[args.volume], slice_index=args.slice, labels=args.labels)

            else:
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(dataloader):
                    display_batch(lr_volumes=lr_volumes, gt_volumes=hr_volumes, plane=args.plane, dimension=args.dim, volume_indexes=volume_indexes(batch_index, args.batch_size), slice_index=args.slice, labels=args.labels)

        elif args.mode == 'compare_scans':
            if int(args.volume) >= 0:
                # Add batch dimension and move inputs to the designated device
                lr_volume = dataset[args.volume][0].unsqueeze(0).to(device)

                display_batch(lr_volumes=lr_volume, sr_volumes=generator_inferer(inputs=lr_volume, network=generator), gt_volumes=[dataset[args.volume][1]], plane=args.plane, dimension=args.dim, volume_indexes=[args.volume], slice_index=args.slice, labels=args.labels)

            else:
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(dataloader):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)

                    display_batch(lr_volumes=lr_volumes, sr_volumes=generator_inferer(inputs=lr_volumes, network=generator), gt_volumes=hr_volumes, plane=args.plane, dimension=args.dim, volume_indexes=volume_indexes(batch_index, args.batch_size), slice_index=args.slice, labels=args.labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Visualizer')
    parser.add_argument('--mode', default=None, help='Display mode {}'.format(modes), type=str)
    parser.add_argument('--lr-dir', help='Path to low-resolution data directory', type=str)
    parser.add_argument('--hr-dir', help='Path to high-resolution data directory', type=str)
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint', type=str)
    parser.add_argument('--augmentation', default=False, help='Apply augmentation techniques to the visualization data', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--pixel-range', default=(None, None), help='Window range of pixel values (min, max). For multi-channel scaling provide ((first_channel_min, ..., last_channel_min), (first_channel_max, ..., last_channel_max))', type=ast.literal_eval)
    parser.add_argument('--plane', default='axial', help='Anatomical planes {}'.format(planes), type=str)
    parser.add_argument('--dim', default='2D', help='Visualization dimension {}'.format(dimensions), type=str)
    parser.add_argument('--volume', default=-1, help='Volume index (-1 for all)', type=int)
    parser.add_argument('--slice', default=-1, help='Slice index (-1 for all)', type=int)
    parser.add_argument('--batch-size', default=4, help='Number of samples to visualize'.format(dimensions), type=int)
    parser.add_argument('--num-workers', default=4, help='Number of subprocesses used for data loading', type=int)
    parser.add_argument('--labels', default=True, help='Display image labels', type=(lambda arg: (str(arg).lower() == 'true')))
    args = parser.parse_args()

    main()

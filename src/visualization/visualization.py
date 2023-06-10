import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.join('..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator
from src.utils.utils import display_batch


# Available display modes
modes = {
    'gt_scans': 'Display ground-truth CT scan counterpart ',
    'lr_scans': 'Display low-resolution CT scan counterpart',
    'sr_scans': 'Display super-resolution CT scan counterpart',
    'gt_lr_scans': 'Display low-resolution and ground truth CT scans jointly',
    'compare_scans': 'Display low-resolution, super-resolution and ground truth CT scans jointly',
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

# Available Target Visualization Tissues
tissues = {
    'full': (-1000, 3000),
    'lung': (-900, -500),
    'fat': (-200, -50),
    'muscle': (-50, 50),
    'bone': (50, 1000)
}


def main():
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Set seed for reproducibility
    torch.manual_seed(20)

    if not (args.checkpoint or args.mode in ('gt_scans', 'lr_scans', 'gt_lr_scans')):
        raise NotImplementedError(f'Provide model checkpoint file to visualize with mode \'{args.mode}\'.')

    if not (args.checkpoint or args.scale_factor) and (args.mode in ('lr_scans', 'gt_lr_scans')):
        raise NotImplementedError(f'Provide model checkpoint file or scale factor to visualize with mode \'{args.mode}\'.')

    if not args.checkpoint:
        # Define Dataset
        dataset = SuperResolutionDataset(data_dir=args.data_dir, scale_factor=args.scale_factor, _min=tissues[args.tissue][0], _max=tissues[args.tissue][1], augmentation=True, downsample=(args.mode != 'gt_scans'))
    else:
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path)

        # Fetch training data statistics
        mean, std = checkpoint['training_data_mean'], checkpoint['training_data_std']

        # Define Dataset
        dataset = SuperResolutionDataset(data_dir=args.data_dir, scale_factor=checkpoint['generator_scale_factor'], _min=tissues[args.tissue][0], _max=tissues[args.tissue][1], mean=mean, std=std, augmentation=True)

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

        # Initialize generator weights
        generator.load_state_dict(checkpoint['generator_state_dict'])

        # Set the model to evaluation mode
        generator.eval()

    # Define Dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)

    with torch.no_grad():
        if args.mode == 'gt_scans':
            if int(args.volume) >= 0:
                display_batch(gt_volumes=[dataset[args.volume][1]], plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

            else:
                for batch_index, (_, hr_volumes, _) in enumerate(dataloader):
                    display_batch(gt_volumes=hr_volumes, plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

        elif args.mode == 'lr_scans':
            if int(args.volume) >= 0:
                display_batch(gt_volumes=[dataset[args.volume][0]], plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

            else:
                for batch_index, (lr_volumes, _, _) in enumerate(dataloader):
                    display_batch(lr_volumes=lr_volumes, plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

        elif args.mode == 'sr_scans':
            if int(args.volume) >= 0:
                display_batch(sr_volumes=[generator(dataset[args.volume][0].unsqueeze(0))], plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

            else:
                for batch_index, (lr_volumes, _, _) in enumerate(dataloader):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)

                    display_batch(sr_volumes=generator(lr_volumes), plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

        elif args.mode == 'gt_lr_scans':
            if int(args.volume) >= 0:
                display_batch(lr_volumes=[dataset[args.volume][0]], gt_volumes=[dataset[args.volume][1]], plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

            else:
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(dataloader):
                    display_batch(lr_volumes=lr_volumes, gt_volumes=hr_volumes, plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

        elif args.mode == 'compare_scans':
            if int(args.volume) >= 0:
                display_batch(lr_volumes=[dataset[args.volume][0]], sr_volumes=[generator(dataset[args.volume][0].unsqueeze(0))], gt_volumes=[dataset[args.volume][1]], plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)

            else:
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(dataloader):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)

                    display_batch(lr_volumes=lr_volumes, sr_volumes=generator(lr_volumes), gt_volumes=hr_volumes, plane=args.plane, dimension=args.dim, slice_index=args.slice, volume_index=args.volume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Visualizer')
    parser.add_argument('--data-dir', required=True, help='Path to data directory', type=str)
    parser.add_argument('--mode', default=None, help='Display mode {}'.format(modes), type=str)
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint', type=str)
    parser.add_argument('--scale-factor', default=None, help='Downsample factor', type=int)
    parser.add_argument('--tissue', default='full', help='Sets Hounsfield Unit (HU) range to visualize tissue {}'.format(tissues.keys()), type=str)
    parser.add_argument('--plane', default='axial', help='anatomical planes {}'.format(planes), type=str)
    parser.add_argument('--dim', default='2D', help='Visualization dimension {}'.format(dimensions), type=str)
    parser.add_argument('--volume', default=-1, help='Volume index (-1 for all)', type=int)
    parser.add_argument('--slice', default=-1, help='Slice index (-1 for all)', type=int)
    parser.add_argument('--batch-size', default=4, help='Number of samples to visualize'.format(dimensions), type=int)
    parser.add_argument('--num-workers', default=4, help='Number of subprocesses used for data loading', type=int)
    args = parser.parse_args()

    main()

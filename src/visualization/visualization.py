import os
import sys
import ast
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.utils.utils import display_batch, set_seed
from src.utils.logger import print_divisor


# Available display modes
modes = {
    'lr_scans': 'Display low-resolution scan counterpart',
    'gen_scans': 'Display generated scan counterpart',
    'gt_scans': 'Display ground truth scan counterpart ',
    'lr_gen_scans': 'Display low-resolution and generated scans jointly',
    'lr_gt_scans': 'Display low-resolution and ground truth scans jointly',
    'gen_gt_scans': 'Display generated and ground truth scans jointly',
    'lr_gen_gt_scans': 'Display low-resolution, generated and ground truth scans jointly',
}

# Available anatomical planes
planes = [
    "axial",
    "coronal",
    "sagittal"
]

# Available Display Dimensions
dimensions = [
    '2D',
    '3D'
]


def main():
    # Set seed for reproducibility
    set_seed(20)

    if args.mode not in modes:
        raise NotImplementedError(f"The mode \'{args.mode}\' is not implemented. Available modes are: {list(modes.keys())}.")

    if not args.lr_dir and args.mode in ('lr_scans', 'lr_gen_scans', 'lr_gt_scans', 'lr_gen_gt_scans'):
        raise NotImplementedError(f'Low-resolution data directory path is required to visualize with mode \'{args.mode}\'.')

    if not args.gen_dir and args.mode in ('gen_scans', 'lr_gen_scans', 'gen_gt_scans', 'lr_gen_gt_scans'):
        raise NotImplementedError(f'Super-resolution data directory path is required to visualize with mode \'{args.mode}\'.')

    if not args.gt_dir and args.mode in ('gt_scans', 'lr_gt_scans', 'gen_gt_scans', 'lr_gen_gt_scans'):
        raise NotImplementedError(f'Ground truth data directory path is required to visualize with mode \'{args.mode}\'.')

    # Define Dataset
    lr_dataset = SuperResolutionDataset(lr_data_dir=args.lr_dir, num_spatial_dimensions=args.num_spatial_dims, _min=args.pixel_range[0], _max=args.pixel_range[1], augmentation=args.augmentation)
    gen_dataset = SuperResolutionDataset(hr_data_dir=args.gen_dir, num_spatial_dimensions=args.num_spatial_dims, _min=args.pixel_range[0], _max=args.pixel_range[1], augmentation=args.augmentation)
    gt_dataset = SuperResolutionDataset(hr_data_dir=args.gt_dir, num_spatial_dimensions=args.num_spatial_dims, _min=args.pixel_range[0], _max=args.pixel_range[1], augmentation=args.augmentation)

    # Define Dataloader
    lr_dataloader = DataLoader(lr_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)
    gen_dataloader = DataLoader(gen_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)
    gt_dataloader = DataLoader(gt_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)

    # Define display configurations. Map each mode to its respective volume types, dataloaders, and batch data indexes
    display_config = {
        'lr_scans': {'volume_types': ['lr_volumes'], 'dataloaders': [lr_dataloader], 'batch_data_indexes': [0]},
        'gen_scans': {'volume_types': ['gen_volumes'], 'dataloaders': [gen_dataloader], 'batch_data_indexes': [1]},
        'gt_scans': {'volume_types': ['gt_volumes'], 'dataloaders': [gt_dataloader], 'batch_data_indexes': [1]},
        'lr_gen_scans': {'volume_types': ['lr_volumes', 'gen_volumes'], 'dataloaders': [lr_dataloader, gen_dataloader], 'batch_data_indexes': [0, 1]},
        'lr_gt_scans': {'volume_types': ['lr_volumes', 'gt_volumes'], 'dataloaders': [lr_dataloader, gt_dataloader], 'batch_data_indexes': [0, 1]},
        'gen_gt_scans': {'volume_types': ['gen_volumes', 'gt_volumes'], 'dataloaders': [gen_dataloader, gt_dataloader], 'batch_data_indexes': [1, 1]},
        'lr_gen_gt_scans': {'volume_types': ['lr_volumes', 'gen_volumes', 'gt_volumes'], 'dataloaders': [lr_dataloader, gen_dataloader, gt_dataloader], 'batch_data_indexes': [0, 1, 1]},
        'batch_metadata_index': 2
    }

    if int(args.volume) >= 0:
        # Define volumes
        volumes = {
            volume_type: dataloader.dataset[args.volume][batch_data_index].unsqueeze(dim=0)
            for volume_type, dataloader, batch_data_index in zip(
                display_config.get(args.mode).get('volume_types'),
                display_config.get(args.mode).get('dataloaders'),
                display_config.get(args.mode).get('batch_data_indexes')
            )
        }

        # Generate list with filename for given volume
        volume_filenames = [random.choice(display_config.get(args.mode).get('dataloaders')).dataset[args.volume][display_config.get('batch_metadata_index')]['filename']]

        display_batch(**volumes, _min=args.pixel_range[0], _max=args.pixel_range[1], plane=args.plane, visualization_dimension=args.visualization_dim, volume_filenames=volume_filenames, slice_index=args.slice, labels=args.labels)
    else:
        for batch_index, batch_datas in enumerate(zip(*display_config.get(args.mode).get('dataloaders'))):
            # Define volumes
            volumes = {
                volume_type: batch_data[batch_data_index]
                for volume_type, batch_data, batch_data_index in zip(
                    display_config.get(args.mode).get('volume_types'),
                    batch_datas,
                    display_config.get(args.mode).get('batch_data_indexes')
                )
            }

            # Generate list with filenames from a given batch
            volume_filenames = random.choice(batch_datas)[display_config.get('batch_metadata_index')]['filename']

            display_batch(**volumes, _min=args.pixel_range[0], _max=args.pixel_range[1], plane=args.plane, visualization_dimension=args.visualization_dim, volume_filenames=volume_filenames, slice_index=args.slice, labels=args.labels)

            print_divisor(width=0, last=True, new_lines_offset=-1) if batch_index + 1 != len(random.choice(display_config.get(args.mode).get('dataloaders'))) else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Visualizer')
    parser.add_argument('--mode', required=True, help='Display mode {}'.format(modes), type=str)
    parser.add_argument('--lr-dir', default=None, help='Path to low-resolution data directory', type=str)
    parser.add_argument('--gen-dir', default=None, help='Path to generated data directory', type=str)
    parser.add_argument('--gt-dir', default=None, help='Path to ground truth data directory', type=str)
    parser.add_argument('--augmentation', default=False, help='Apply augmentation techniques to the visualization data', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--pixel-range', default=(None, None), help='Window range of pixel values (min, max). For multi-channel scaling provide ((first_channel_min, ..., last_channel_min), (first_channel_max, ..., last_channel_max))', type=lambda arg: np.array(ast.literal_eval(arg)))
    parser.add_argument('--num-spatial-dims', default=3, help='Number of spatial dimensions', type=int)
    parser.add_argument('--plane', default='axial', help='Anatomical planes {}'.format(planes), type=str)
    parser.add_argument('--visualization-dim', default='2D', help='Visualization dimension {}'.format(dimensions), type=str)
    parser.add_argument('--volume', default=-1, help='Volume index (-1 for all)', type=int)
    parser.add_argument('--slice', default=-1, help='Slice index (-1 for all)', type=int)
    parser.add_argument('--batch-size', default=2, help='Number of samples to visualize'.format(dimensions), type=int)
    parser.add_argument('--num-workers', default=4, help='Number of subprocesses used for data loading', type=int)
    parser.add_argument('--labels', default=True, help='Display image labels', type=(lambda arg: (str(arg).lower() == 'true')))
    args = parser.parse_args()

    main()

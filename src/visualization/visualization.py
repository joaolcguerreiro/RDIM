import argparse


# Available display modes
modes = {
    'display_ct_scan': 'Display CT scan without processing.',
    'display_lr_ct_scan': 'Display CT scan low-resolution counterpart.',
    'display_sr_ct_scan': 'Display CT scan super-resolution counterpart.',
    'display_all_ct_scan': 'Display low-resolution, super-resolution and ground truth CT scans side by side.',
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
    # Load model
    if args.path:
        # model = ...

    # Define dataloder....
    # dataloader = ...


    # CHECK IF VOLUME IS PASSED IN ARGS. IF NOT LOOPS OVER DATA WITH DATALOADER.
    # MAYBE WE NEED TO CHANGE SPLIT_DATA TO RENAME FILES IN ORDER TO BE ABLE TO INDEX THEM

    # display_volumes(-1, -1, "axial", [volume_to_grayscale(get_volume_data(volume_dir))], 1, 1, '2D')
    pass

    elif args.mode == 'display_sr_ct_scan':
        if not args.path:
            raise NotImplementedError('Provide the model checkpoint path to super-resolve the ct scans.')

        ...

    elif args.mode == 'display_all_ct_scan':
        if not args.path:
            raise NotImplementedError('Provide the model checkpoint path to super-resolve the ct scans.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Visualizer')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint', type=str)
    parser.add_argument('--dir', help='Path to data directory', type=str)
    parser.add_argument('--plane', default='axial', help='anatomical planes {}'.format(planes), type=str)
    parser.add_argument('--mode', default=None, help='Display mode {}'.format(modes), type=str)
    parser.add_argument('--volume', default=-1, help='Volume index (-1 for all)', type=int)
    parser.add_argument('--slice', default=-1, help='Slice index (-1 for all)', type=int)
    parser.add_argument('--dim', default='2D', help='Visualization dimension {}'.format(dimensions), type=str)
    args = parser.parse_args()
    main()

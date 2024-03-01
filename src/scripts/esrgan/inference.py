import os
import sys
import ast
import argparse
import datetime
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from monai.inferers.inferer import SlidingWindowInferer
sys.path.append(os.path.join('..', '..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator
from src.utils.utils import clamp, create_experiment_directory, save_samples, save_batch, set_seed
from src.utils.logger import setup_logging, print_divisor


def main():
    # Set seed for reproducibility
    set_seed(20)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Fetch training data statistics
    _min, _max, mean, std = checkpoint['training_data_min'], checkpoint['training_data_max'], checkpoint['training_data_mean'], checkpoint['training_data_std']

    # Define dataset and data loader
    inference_dataset = SuperResolutionDataset(lr_data_dir=args.lr_dir, scale_factor=checkpoint['generator_scale_factor'], _min=_min, _max=_max, mean=mean, std=std, augmentation=False)
    inference_dataloader = DataLoader(inference_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)

    # Define model
    generator = Generator(
        in_channels=checkpoint['generator_in_channels'],
        out_channels=checkpoint['generator_out_channels'],
        num_features=checkpoint['generator_num_features'],
        num_rrdb_blocks=checkpoint['generator_num_rrdb_blocks'],
        num_rrdb_db_blocks=checkpoint['generator_num_rrdb_db_blocks'],
        num_rrdb_db_conv_blocks=checkpoint['generator_num_rrdb_db_conv_blocks'],
        num_rrdb_db_growth_features=checkpoint['generator_num_rrdb_db_growth_features'],
        scale_factor=checkpoint['generator_scale_factor'],
    ).to(device)

    # Log Generator
    summary(generator, input_size=(1, inference_dataset.get_num_channels(), *checkpoint['training_lr_patch_size']), col_names=("input_size", "output_size", "num_params"), depth=1, device=device)
    print_divisor(width=0, last=True, new_lines_offset=-1)

    # Define patch-based inferer
    generator_inferer = SlidingWindowInferer(roi_size=checkpoint['training_lr_patch_size'], overlap=0.25, mode='gaussian', sw_device=device, device=device)

    # Initialize generator weights
    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Inference loop
    print_divisor()

    # Set the model to evaluation mode
    generator.eval()

    # Define inference progress bar
    inference_pbar = tqdm(inference_dataloader, desc="Inferring")

    with torch.no_grad():
        for lr_volumes, _, sr_volumes_metadata in inference_pbar:
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)

            # Forward pass
            gen_volumes = generator_inferer(inputs=lr_volumes, network=generator)

            # Clamp generated data
            gen_volumes = torch.stack([clamp(gen_volume, _min=_min, _max=_max, mean=mean, std=std) for gen_volume in gen_volumes]) if args.clamp else gen_volumes

            # Plot sample figures
            save_samples(
                lr_volumes=torch.stack([clamp(lr_volume, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std) for lr_volume in lr_volumes]),
                gen_volumes=torch.stack([clamp(gen_volume, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std) for gen_volume in gen_volumes]),
                bins=100,
                _min=(args.sample_pixel_range[0] or _min), _max=(args.sample_pixel_range[1] or _max), mean=mean, std=std,
                save_dir_path=os.path.join(experiment_path, 'figures', 'samples'),
                volumes_metadata=sr_volumes_metadata,
                fig_size=(3, 3)
            )

            # Save volumes
            save_batch(gen_volumes, sr_volumes_metadata, os.path.join(experiment_path, 'results'), mean=mean, std=std)

        # Logging the inference progress bar
        logging.info(inference_pbar)

    print_divisor()


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint', type=str)
    parser.add_argument('--lr-dir', required=True, help='Path to low-resolution data directory', type=str)
    parser.add_argument('--save-dir', required=True, help='Path to directory where results are saved', type=str)
    parser.add_argument('--clamp', default=True, help='Clamp generated data to original data range', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--sample-pixel-range', default=(None, None), help='Window range of pixel values (min, max). For multi-channel scaling provide ((first_channel_min, ..., last_channel_min), (first_channel_max, ..., last_channel_max))', type=lambda arg: np.array(ast.literal_eval(arg)))
    parser.add_argument('--batch-size', default=2, help='Number of samples per batch', type=int)
    parser.add_argument('--num-workers', default=4, help='Number of subprocesses used for data loading', type=int)
    args = parser.parse_args()

    # Define experiment folder
    experiment_path = os.path.join(args.save_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_path = args.checkpoint

    # Create experiment folder
    create_experiment_directory(experiment_path=experiment_path, checkpoint_path=checkpoint_path, inference_mode=True)

    # Set up logging
    setup_logging(os.path.join(experiment_path, 'logs', 'inference.log'))

    main()

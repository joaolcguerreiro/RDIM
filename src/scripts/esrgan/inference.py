import os
import sys
import argparse
import logging
import torch
import datetime
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
sys.path.append(os.path.join('..', '..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator
from src.utils.utils import save_batch, compute_batch_super_resolution_metadata
from src.utils.logger import setup_logger


def main():
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Set seed for reproducibility
    torch.manual_seed(20)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Fetch training data statistics
    _min, _max, mean, std = checkpoint['training_data_min'], checkpoint['training_data_max'], checkpoint['training_data_mean'], checkpoint['training_data_std']

    inference_dataset = SuperResolutionDataset(data_dir=args.lr_dir, _min=_min, _max=_max, mean=mean, std=std, augmentation=False, downsample=False)
    inference_dataloader = DataLoader(inference_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=SuperResolutionDataset.sr_dataset_collate)

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

    # Log Generator
    summary(generator, input_size=(1, *inference_dataset.get_low_resolution_shape()), depth=1, device=device)

    # Initialize generator weights
    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Set the model to evaluation mode
    generator.eval()

    # Define validation progress bar
    inference_pbar = tqdm(inference_dataloader, desc="Inferring")

    with torch.no_grad():
        for batch_index, (lr_volumes, _, lr_metadata) in enumerate(inference_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)

            # Forward pass
            gen_volumes = generator(lr_volumes)

            # Save volumes
            save_batch(gen_volumes, compute_batch_super_resolution_metadata(lr_metadata, checkpoint['generator_scale_factor']), batch_index, os.path.join(args.save_dir, 'results'), mean=mean, std=std)

        # Logging the validation progress bar
        logging.info(inference_pbar)


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint', type=str)
    parser.add_argument('--lr-dir', required=True, help='Path to low-resolution data directory', type=str)
    parser.add_argument('--save-dir', required=True, help='Path to directory where results are saved', type=str)
    parser.add_argument('--batch-size', default=4, help='Number of samples per batch', type=int)
    parser.add_argument('--num-workers', default=4, help='Number of subprocesses used for data loading', type=int)
    args = parser.parse_args()

    # Define experiment folder
    experiment_path = os.path.join(args.save_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_path = args.checkpoint

    # Set up logger
    setup_logger(os.path.join(experiment_path, 'inference.log'))

    main()
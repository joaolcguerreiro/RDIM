import os
import sys
import ast
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.inferer import MultiScaleConditionalSlidingWindowInferer
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.rdim import GaussianDiffusion, NoisePredictor, LREncoder
from src.utils.metrics import SuperResolutionMetrics
from src.utils.utils import parse_patch_size, clamp, save_samples, save_batch, collect_batch_pixels, parse_bin_range, compute_max_frequency, draw_box_plot, draw_histogram, set_seed
from src.utils.logger import setup_logging, print_divisor


def main():
    # Set seed for reproducibility
    set_seed(20)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Fetch training data statistics
    _min, _max, mean, std = checkpoint['training_data_min'], checkpoint['training_data_max'], checkpoint['training_data_mean'], checkpoint['training_data_std']

    # Define dataset and data loader
    test_dataset = SuperResolutionDataset(lr_data_dir=config["DATA"]["TESTING"]["LR_DIR_PATH"], hr_data_dir=config["DATA"]["TESTING"]["HR_DIR_PATH"], scale_factor=checkpoint['gaussian_diffusion_scale_factor'], _min=_min, _max=_max, mean=mean, std=std, augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["TESTING"]["BATCH_SIZE"], num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate)

    # Define models
    gaussian_diffusion = GaussianDiffusion(
        gamma=checkpoint['gaussian_diffusion_gamma'],
        eta=checkpoint['gaussian_diffusion_eta'],
        num_timesteps=checkpoint['gaussian_diffusion_num_timesteps'],
        num_sampling_timesteps=(args.num_sampling_timesteps or checkpoint['gaussian_diffusion_num_sampling_timesteps']),
        sampling_timestep_schedule=(args.sampling_timestep_schedule or checkpoint['gaussian_diffusion_sampling_timestep_schedule']),
        beta_schedule=checkpoint['gaussian_diffusion_beta_schedule'],
        lr_condition_encoding=checkpoint['gaussian_diffusion_lr_condition_encoding'],
        scale_factor=checkpoint['gaussian_diffusion_scale_factor'],
        **{key.lower(): value for key, value in checkpoint['gaussian_beta_schedule_parameters'].items()}
    ).to(device)

    noise_predictor = NoisePredictor(
        in_channels=checkpoint['noise_predictor_in_channels'],
        out_channels=checkpoint['noise_predictor_out_channels'],
        num_cond_features=(checkpoint['lr_encoder_num_features'] if checkpoint['gaussian_diffusion_lr_condition_encoding'] else checkpoint['noise_predictor_in_channels']),
        num_inner_features=checkpoint['noise_predictor_num_inner_features'],
        feature_multipliers=checkpoint['noise_predictor_feature_multipliers'],
        num_rb_blocks=checkpoint['noise_predictor_num_rb_blocks'],
        num_rb_conv_blocks=checkpoint['noise_predictor_num_rb_conv_blocks'],
        attention_levels=checkpoint['noise_predictor_attention_levels'],
        num_attention_heads=checkpoint['noise_predictor_num_attention_heads'],
        num_groups=checkpoint['noise_predictor_num_groups'],
        scale_factor=checkpoint['gaussian_diffusion_scale_factor'],
    ).to(device)

    if checkpoint['gaussian_diffusion_lr_condition_encoding']:
        lr_encoder = LREncoder(
            in_channels=checkpoint['lr_encoder_in_channels'],
            out_channels=checkpoint['lr_encoder_out_channels'],
            num_features=checkpoint['lr_encoder_num_features'],
            num_rrdb_blocks=checkpoint['lr_encoder_num_rrdb_blocks'],
            num_rrdb_db_blocks=checkpoint['lr_encoder_num_rrdb_db_blocks'],
            num_rrdb_db_conv_blocks=checkpoint['lr_encoder_num_rrdb_db_conv_blocks'],
            num_rrdb_db_growth_features=checkpoint['lr_encoder_num_rrdb_db_growth_features'],
            scale_factor=checkpoint['gaussian_diffusion_scale_factor'],
        ).to(device)

    # Log Noise Predictor and LR Encoder
    summary(noise_predictor, input_data=[torch.randn(1, checkpoint['noise_predictor_in_channels'], *map(lambda dim: dim * checkpoint['gaussian_diffusion_scale_factor'], (lr_patch_size if (lr_patch_size := parse_patch_size(args.lr_patch_size, test_dataset.get_low_resolution_shape(only_spatial_dims=True))) is not None else checkpoint['training_lr_patch_size']))), torch.randn(1, (checkpoint['lr_encoder_num_features'] if checkpoint['gaussian_diffusion_lr_condition_encoding'] else checkpoint['noise_predictor_in_channels']), *(lr_patch_size if (lr_patch_size := parse_patch_size(args.lr_patch_size, test_dataset.get_low_resolution_shape(only_spatial_dims=True))) is not None else checkpoint['training_lr_patch_size'])), gaussian_diffusion.sample_timesteps(1)], col_names=("input_size", "output_size", "num_params"), depth=1, device=device)
    print_divisor(width=0, last=True, new_lines_offset=-1)

    if checkpoint['gaussian_diffusion_lr_condition_encoding']:
        summary(lr_encoder, input_size=(1, checkpoint['lr_encoder_in_channels'], *(lr_patch_size if (lr_patch_size := parse_patch_size(args.lr_patch_size, test_dataset.get_low_resolution_shape(only_spatial_dims=True))) is not None else checkpoint['training_lr_patch_size'])), col_names=("input_size", "output_size", "num_params"), depth=1, device=device)
        print_divisor(width=0, last=True, new_lines_offset=-1)

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1 or torch.mps.device_count() > 1:
        noise_predictor = nn.DataParallel(noise_predictor, device_ids=config["DATA"]["GPU_IDS"])

        if checkpoint['gaussian_diffusion_lr_condition_encoding']:
            lr_encoder = nn.DataParallel(lr_encoder, device_ids=config["DATA"]["GPU_IDS"])

    # Define patch-based inferer
    noise_predictor_inferer = MultiScaleConditionalSlidingWindowInferer(roi_size=list(map(lambda dim: dim * checkpoint['gaussian_diffusion_scale_factor'], (args.lr_patch_size if args.lr_patch_size is not None else checkpoint['training_lr_patch_size']))), scale_factor=checkpoint['gaussian_diffusion_scale_factor'], overlap=0.125, mode='gaussian', sw_batch_size=config["TESTING"]["PATCH_SLIDING_WINDOW_BATCH_SIZE"], sw_device=device, device=device)

    # Define metrics
    metrics = SuperResolutionMetrics(num_spatial_dims=len(checkpoint['training_lr_patch_size']), _min=_min, _max=_max, mean=mean, std=std)

    # Initialize models weights
    (noise_predictor.module if (torch.cuda.device_count() > 1 or torch.mps.device_count() > 1) else noise_predictor).load_state_dict(checkpoint['noise_predictor_state_dict'])
    (lr_encoder.module if (torch.cuda.device_count() > 1 or torch.mps.device_count() > 1) else lr_encoder).load_state_dict(checkpoint['lr_encoder_state_dict']) if checkpoint['gaussian_diffusion_lr_condition_encoding'] else None

    # Testing loop
    print_divisor()

    # Set the noise predictor to evaluation mode
    noise_predictor.eval()
    hr_pixel_values = []
    gen_pixel_values = []

    # Set the LR encoder to evaluation mode if applicable
    if checkpoint['gaussian_diffusion_lr_condition_encoding']:
        lr_encoder.eval()

    # Define testing progress bar
    test_pbar = tqdm(test_dataloader, desc="Testing")

    with torch.no_grad():
        for batch_index, (lr_volumes, hr_volumes, sr_volumes_metadata) in enumerate(test_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)
            hr_volumes = hr_volumes.to(device)

            # Reverse Diffusion (Forward pass xT)
            gen_volumes = gaussian_diffusion.sample(lr_volumes, noise_predictor, (lr_encoder if checkpoint['gaussian_diffusion_lr_condition_encoding'] else None), noise_predictor_inferer)

            # Clamp generated data
            gen_volumes = torch.stack([clamp(gen_volume, _min=_min, _max=_max, mean=mean, std=std) for gen_volume in gen_volumes]) if args.clamp else gen_volumes

            # Compute metrics
            scores = metrics.compute_scores(gen_volumes.to('cpu'), hr_volumes.to('cpu'))
            test_pbar.set_postfix(scores)

            # Plot sample figures
            save_samples(
                lr_volumes=torch.stack([clamp(lr_volume, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std) for lr_volume in lr_volumes]),
                gen_volumes=torch.stack([clamp(gen_volume, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std) for gen_volume in gen_volumes]),
                gt_volumes=torch.stack([clamp(hr_volume, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std) for hr_volume in hr_volumes]),
                bins=100,
                _min=(args.sample_pixel_range[0] if args.sample_pixel_range[0] is not None else _min), _max=(args.sample_pixel_range[1] if args.sample_pixel_range[1] is not None else _max), mean=mean, std=std,
                save_dir_path=os.path.join(experiment_path, 'figures', 'test', 'samples'),
                volumes_metadata=sr_volumes_metadata,
                fig_size=(3, 3)
            )

            # Save volumes
            save_batch(gen_volumes, sr_volumes_metadata, os.path.join(experiment_path, 'results'), mean=mean, std=std)

            if batch_index + 1 == len(test_pbar):
                test_pbar.set_postfix()

                # Track last batch pixel intensities
                hr_pixel_values = collect_batch_pixels(hr_volumes, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std)
                gen_pixel_values = collect_batch_pixels(gen_volumes, _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std)

        # Logging the testing progress bar
        logging.info(test_pbar)

    print_divisor(last=True)

    # Log metrics
    metrics.summary()

    # Plot metric scores
    for metric_name, metric_scores in metrics.get_scores().items():
        draw_box_plot(
            data=[
                {'Model': ['SRMeDiff'] * len(metric_scores), 'Scale factor': [checkpoint['gaussian_diffusion_scale_factor']] * len(metric_scores), metric_name: metric_scores}
            ],
            title='Quantitative Results',
            x_label='Scale factor',
            y_label=metric_name,
            hue='Model',
            save_filename=os.path.join(experiment_path, 'figures', 'test', f'{metric_name}', f'{args.epoch}')
        )

    # Plot sample pixel distribution
    draw_histogram(
        data=[
            {'Pixel Intensity': gen_pixel_values, 'Data Type': ['Reconstruction'] * len(gen_pixel_values)},
            {'Pixel Intensity': hr_pixel_values, 'Data Type': ['Ground Truth'] * len(hr_pixel_values)}
        ],
        title='Sample Pixel Distribution',
        x_label='Pixel Intensity',
        hue='Data Type',
        bins=100,
        bin_range=parse_bin_range(_min=(args.sample_pixel_range[0] if args.sample_pixel_range[0] is not None else _min), _max=(args.sample_pixel_range[1] if args.sample_pixel_range[1] is not None else _max)),
        frequency_range=(None, compute_max_frequency(pixel_values=hr_pixel_values, bins=100, _min=(args.sample_pixel_range[0] if args.sample_pixel_range[0] is not None else _min), _max=(args.sample_pixel_range[1] if args.sample_pixel_range[1] is not None else _max)) * 1.1),
        element='poly',
        fill=False,
        save_filename=os.path.join(experiment_path, 'figures', 'test', 'pixel_distribution', f'{args.epoch}')
    )


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Testing Script')
    parser.add_argument('--exp', required=True, help="Path to experiment directory", type=str)
    parser.add_argument('--epoch', required=True, help="Epoch to be tested", type=str)
    parser.add_argument('--clamp', default=True, help='Clamp generated data to original data range', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--sample-pixel-range', default=(None, None), help='Window range of pixel values (min, max). For multi-channel scaling provide ((first_channel_min, ..., last_channel_min), (first_channel_max, ..., last_channel_max))', type=lambda arg: np.array(ast.literal_eval(arg)))
    parser.add_argument('--lr-patch-size', default=None, help="Override low-resolution training patch size", type=ast.literal_eval)
    parser.add_argument('--num-sampling-timesteps', default=None, help="Override number of timesteps in the reverse trajectory", type=int)
    parser.add_argument('--sampling-timestep-schedule', default=None, help="Override timestep scheduler for the reverse trajectory", type=str)
    args = parser.parse_args()

    # Define experiment folder
    experiment_path = args.exp
    config_path = os.path.join(args.exp, "configs", "config.yaml")
    checkpoint_path = os.path.join(args.exp, "checkpoints", f"{args.epoch}.pth")

    # Load configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logging
    setup_logging(os.path.join(experiment_path, "logs", "test", f'{args.epoch}.log'))

    main()

import os
import sys
import ast
import argparse
import datetime
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.inferer import MultiScaleConditionalSlidingWindowInferer
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.voxrdiff import GaussianDiffusion, NoisePredictor
from src.models.voxrdiff import LREncoder
from src.utils.losses import VGGLoss, SSIMLoss
from src.utils.utils import upsample, clamp, denormalize, create_experiment_directory, compute_data_statistics, draw_line_chart, save_sample, set_seed
from src.utils.logger import setup_logging, print_divisor


def main():
    # Set seed for reproducibility
    set_seed(20)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Compute training data statistics
    _min, _max, mean, std = compute_data_statistics(data_dir=config["DATA"]["TRAINING"]["HR_DIR_PATH"], num_channels=config["MODEL"]["NOISE_PREDICTOR"]["IN_CHANNELS"])

    # Define datasets and data loaders
    train_dataset = SuperResolutionDataset(lr_data_dir=config["DATA"]["TRAINING"]["LR_DIR_PATH"], hr_data_dir=config["DATA"]["TRAINING"]["HR_DIR_PATH"], scale_factor=config["MODEL"]["SCALE_FACTOR"], lr_patch_size=config["TRAINING"]["LR_PATCH_SIZE"], _min=_min, _max=_max, mean=mean, std=std, repeat_sample_factor=config["TRAINING"]["REPEAT_SAMPLE_FACTOR"], augmentation=config["TRAINING"]["AUGMENTATION"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate)
    val_dataset = SuperResolutionDataset(lr_data_dir=config["DATA"]["VALIDATION"]["LR_DIR_PATH"], hr_data_dir=config["DATA"]["VALIDATION"]["HR_DIR_PATH"], _min=_min, _max=_max, mean=mean, std=std, augmentation=False) if args.val else None
    val_dataloader = DataLoader(val_dataset, batch_size=config["VALIDATION"]["BATCH_SIZE"], num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate) if args.val else None
    pre_train_lr_encoder_dataset = SuperResolutionDataset(lr_data_dir=config["DATA"]["TRAINING"]["LR_DIR_PATH"], hr_data_dir=config["DATA"]["TRAINING"]["HR_DIR_PATH"], scale_factor=config["MODEL"]["SCALE_FACTOR"], lr_patch_size=config["TRAINING"]["LR_PATCH_SIZE"], _min=_min, _max=_max, mean=mean, std=std, repeat_sample_factor=config["PRE_TRAINING_LR_ENCODER"]["REPEAT_SAMPLE_FACTOR"], augmentation=config["TRAINING"]["AUGMENTATION"]) if config["MODEL"]["LR_CONDITION_ENCODING"] else None
    pre_train_lr_encoder_dataloader = DataLoader(pre_train_lr_encoder_dataset, batch_size=config["PRE_TRAINING_LR_ENCODER"]["BATCH_SIZE"], shuffle=True, num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate) if config["MODEL"]["LR_CONDITION_ENCODING"] else None

    # Define models
    gaussian_diffusion = GaussianDiffusion(
        num_timesteps=config["MODEL"]["NUM_TIMESTEPS"],
        beta_schedule=config["MODEL"]["BETA_SCHEDULE"],
        residual_prediction=config["MODEL"]["RESIDUAL_PREDICTION"],
        lr_condition_encoding=config["MODEL"]["LR_CONDITION_ENCODING"],
        scale_factor=config["MODEL"]["SCALE_FACTOR"],
        **{key.lower(): value for key, value in config["BETA_SCHEDULE_PARAMETERS"][config["MODEL"]["BETA_SCHEDULE"]].items()}
    ).to(device)

    noise_predictor = NoisePredictor(
        in_channels=config["MODEL"]["NOISE_PREDICTOR"]["IN_CHANNELS"],
        out_channels=config["MODEL"]["NOISE_PREDICTOR"]["OUT_CHANNELS"],
        num_cond_features=(config["MODEL"]["LR_ENCODER"]["NUM_FEATURES"] if config["MODEL"]["LR_CONDITION_ENCODING"] else config["MODEL"]["NOISE_PREDICTOR"]["IN_CHANNELS"]),
        num_inner_features=config["MODEL"]["NOISE_PREDICTOR"]["NUM_INNER_FEATURES"],
        feature_multipliers=config["MODEL"]["NOISE_PREDICTOR"]["FEATURE_MULTIPLIERS"],
        num_rdb_blocks=config["MODEL"]["NOISE_PREDICTOR"]["NUM_RDB_BLOCKS"],
        num_rdb_conv_blocks=config["MODEL"]["NOISE_PREDICTOR"]["NUM_RDB_CONV_BLOCKS"],
        num_rdb_growth_features=config["MODEL"]["NOISE_PREDICTOR"]["NUM_RDB_GROWTH_FEATURES"],
        attention_levels=config["MODEL"]["NOISE_PREDICTOR"]["ATTENTION_LEVELS"],
        num_attention_heads=config["MODEL"]["NOISE_PREDICTOR"]["NUM_ATTENTION_HEADS"],
        num_groups=config["MODEL"]["NOISE_PREDICTOR"]["NUM_GROUPS"],
        scale_factor=config["MODEL"]["SCALE_FACTOR"]
    ).to(device)

    if config["MODEL"]["LR_CONDITION_ENCODING"]:
        lr_encoder = LREncoder(
            in_channels=config["MODEL"]["LR_ENCODER"]["IN_CHANNELS"],
            out_channels=config["MODEL"]["LR_ENCODER"]["OUT_CHANNELS"],
            num_features=config["MODEL"]["LR_ENCODER"]["NUM_FEATURES"],
            num_rrdb_blocks=config["MODEL"]["LR_ENCODER"]["NUM_RRDB_BLOCKS"],
            num_rrdb_db_blocks=config["MODEL"]["LR_ENCODER"]["NUM_RRDB_DB_BLOCKS"],
            num_rrdb_db_conv_blocks=config["MODEL"]["LR_ENCODER"]["NUM_RRDB_DB_CONV_BLOCKS"],
            num_rrdb_db_growth_features=config["MODEL"]["LR_ENCODER"]["NUM_RRDB_DB_GROWTH_FEATURES"],
            scale_factor=config["MODEL"]["SCALE_FACTOR"]
        ).to(device)

        # Replace forward method to extract features
        lr_encoder_forward_method = lr_encoder.forward
        lr_encoder.forward = lr_encoder.extract_features

    # Initialize weights
    noise_predictor.initialize_weights()
    lr_encoder.initialize_weights() if config["MODEL"]["LR_CONDITION_ENCODING"] else None

    # Log Noise Predictor and LR Encoder
    summary(noise_predictor, input_data=[torch.randn(1, *train_dataset.get_high_resolution_shape()), torch.randn(1, (config["MODEL"]["LR_ENCODER"]["NUM_FEATURES"] if config["MODEL"]["LR_CONDITION_ENCODING"] else config["MODEL"]["NOISE_PREDICTOR"]["IN_CHANNELS"]), *train_dataset.get_low_resolution_shape(only_spatial_dims=True)), gaussian_diffusion.sample_timesteps(1)], col_names=("input_size", "output_size", "num_params"), depth=1, device=device)
    print_divisor(width=0, last=True, new_lines_offset=-1)

    if config["MODEL"]["LR_CONDITION_ENCODING"]:
        summary(lr_encoder, input_size=(1, *train_dataset.get_low_resolution_shape()), col_names=("input_size", "output_size", "num_params"), depth=1, device=device)
        print_divisor(width=0, last=True, new_lines_offset=-1)

    # Define patch-based inferers
    noise_predictor_inferer = MultiScaleConditionalSlidingWindowInferer(roi_size=train_dataset.get_high_resolution_shape(only_spatial_dims=True), scale_factor=config["MODEL"]["SCALE_FACTOR"], overlap=0.125, mode='gaussian', sw_batch_size=config["VALIDATION"]["PATCH_SLIDING_WINDOW_BATCH_SIZE"], sw_device=device, device=device)

    # Define optimizer and scheduler
    noise_predictor_optimizer = optim.Adam(
        list(noise_predictor.parameters()) + (list(lr_encoder.parameters()) if not config["TRAINING"]["FIX_LR_ENCODER"] and config["MODEL"]["LR_CONDITION_ENCODING"] else []),
        lr=config['OPTIMIZER']['NOISE_PREDICTOR']['LEARNING_RATE'],
        betas=(config['OPTIMIZER']['NOISE_PREDICTOR']['BETA_1'], config['OPTIMIZER']['NOISE_PREDICTOR']['BETA_2']),
        weight_decay=config['OPTIMIZER']['NOISE_PREDICTOR']['WEIGHT_DECAY']
    )
    noise_predictor_scheduler = optim.lr_scheduler.CosineAnnealingLR(noise_predictor_optimizer, T_max=(len(train_dataloader) * config['TRAINING']['EPOCHS']), eta_min=config['SCHEDULER']['NOISE_PREDICTOR']['ETA_MIN'])

    if config["MODEL"]["LR_CONDITION_ENCODING"]:
        pre_train_lr_encoder_optimizer = optim.Adam(
            lr_encoder.parameters(),
            lr=config['OPTIMIZER']['PRE_TRAIN_LR_ENCODER']['LEARNING_RATE'],
            betas=(config['OPTIMIZER']['PRE_TRAIN_LR_ENCODER']['BETA_1'], config['OPTIMIZER']['PRE_TRAIN_LR_ENCODER']['BETA_2']),
            weight_decay=config['OPTIMIZER']['PRE_TRAIN_LR_ENCODER']['WEIGHT_DECAY']
        )
        pre_train_lr_encoder_scheduler = optim.lr_scheduler.StepLR(pre_train_lr_encoder_optimizer, step_size=config['SCHEDULER']['PRE_TRAIN_LR_ENCODER']['STEP_SIZE'], gamma=config['SCHEDULER']['PRE_TRAIN_LR_ENCODER']['GAMMA'])

    # Define the loss function
    diffusion_criterion = nn.MSELoss().to(device)
    perceptual_criterion = VGGLoss(_min=_min, _max=_max, mean=mean, std=std, patch_size=train_dataset.get_high_resolution_shape(only_spatial_dims=True)).to(device) if config['PRE_TRAINING_LR_ENCODER']['PERCEPTUAL_LOSS_VGG'] else SSIMLoss(num_spatial_dims=len(train_dataset.get_low_resolution_shape(only_spatial_dims=True)), _min=_min, _max=_max, mean=mean, std=std).to(device)
    content_criterion = nn.L1Loss().to(device)

    # Define number of epochs and resume training if applicable
    if args.resume:
        # Load the saved checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Initialize weights, optimizers and epoch count
        noise_predictor.load_state_dict(checkpoint['noise_predictor_state_dict'])
        noise_predictor_optimizer.load_state_dict(checkpoint['noise_predictor_optimizer_state_dict'])
        noise_predictor_scheduler.load_state_dict(checkpoint['noise_predictor_scheduler_state_dict'])
        lr_encoder.load_state_dict(checkpoint['lr_encoder_state_dict']) if config["MODEL"]["LR_CONDITION_ENCODING"] else None
        start_epoch = checkpoint['epoch'] + 1
        train_dataset.update_epoch(start_epoch)
        val_dataset.update_epoch(start_epoch) if args.val else None
        pre_train_lr_encoder_start_epoch = config['PRE_TRAINING_LR_ENCODER']['EPOCHS'] if config["MODEL"]["LR_CONDITION_ENCODING"] else -1
        noise_predictor_train_losses = checkpoint['noise_predictor_train_losses']
        noise_predictor_val_losses = checkpoint['noise_predictor_val_losses']
    else:
        # Load pre-trained models
        if config["MODEL"]["NOISE_PREDICTOR"]["PRE_TRAINED_PATH"]:
            # Load the pre-trained checkpoint
            checkpoint = torch.load(config["MODEL"]["NOISE_PREDICTOR"]["PRE_TRAINED_PATH"])

            # Initialize noise predictor weights
            noise_predictor.load_state_dict(checkpoint['noise_predictor_state_dict'])

        if config["MODEL"]["LR_ENCODER"]["PRE_TRAINED_PATH"] and config["MODEL"]["LR_CONDITION_ENCODING"]:
            # Load the pre-trained checkpoint
            checkpoint = torch.load(config["MODEL"]["LR_ENCODER"]["PRE_TRAINED_PATH"])

            # Initialize LR Encoder weights
            lr_encoder.load_state_dict(checkpoint['lr_encoder_state_dict'])

        start_epoch = 0
        pre_train_lr_encoder_start_epoch = 0
        noise_predictor_train_losses = dict()
        noise_predictor_val_losses = dict()

    end_epoch = config['TRAINING']['EPOCHS']
    pre_train_lr_encoder_end_epoch = config['PRE_TRAINING_LR_ENCODER']['EPOCHS']

    # Pre-train LR encoder loop
    if not args.resume and config["MODEL"]["LR_CONDITION_ENCODING"]:
        print_divisor()

        # Replace extract features method with forward
        lr_encoder.forward = lr_encoder_forward_method

        for pre_train_epoch in range(pre_train_lr_encoder_start_epoch, pre_train_lr_encoder_end_epoch):
            # Set model to train mode
            lr_encoder.train()
            lr_encoder_train_loss = 0.0

            # Define training progress bar
            pre_train_lr_encoder_pbar = tqdm(pre_train_lr_encoder_dataloader, desc=f"Epoch {pre_train_epoch + 1}/{pre_train_lr_encoder_end_epoch}")

            for batch_index, (lr_volumes, hr_volumes, _) in enumerate(pre_train_lr_encoder_pbar):
                # Move inputs to the designated device
                lr_volumes = lr_volumes.to(device)
                hr_volumes = hr_volumes.to(device)

                # Zero the gradients
                pre_train_lr_encoder_optimizer.zero_grad()

                # Forward pass
                gen_volumes = lr_encoder(lr_volumes)

                # Compute perceptual loss
                perceptual_loss = perceptual_criterion(gen_volumes, hr_volumes) if config['LOSS_WEIGHTS']['PRE_TRAIN_LR_ENCODER']['PERCEPTUAL_LOSS'] != 0.0 else 0.0

                # Compute pixel-wise loss
                content_loss = content_criterion(gen_volumes, hr_volumes) if config['LOSS_WEIGHTS']['PRE_TRAIN_LR_ENCODER']['CONTENT_LOSS'] != 0.0 else 0.0

                # LR Encoder Total loss
                lr_encoder_total_loss = config['LOSS_WEIGHTS']['PRE_TRAIN_LR_ENCODER']['PERCEPTUAL_LOSS'] * perceptual_loss + config['LOSS_WEIGHTS']['PRE_TRAIN_LR_ENCODER']['CONTENT_LOSS'] * content_loss

                # Backward pass
                lr_encoder_total_loss.backward()
                pre_train_lr_encoder_optimizer.step()

                # Update the training loss
                lr_encoder_train_loss += lr_encoder_total_loss.item()
                pre_train_lr_encoder_pbar.set_postfix({
                    'LR Enc. Train. Loss': f'{lr_encoder_total_loss.item():.4f}',
                })

                # Compute average training loss for this epoch
                if batch_index + 1 == len(pre_train_lr_encoder_pbar):
                    lr_encoder_train_loss /= len(pre_train_lr_encoder_dataloader)
                    pre_train_lr_encoder_pbar.set_postfix({
                        "Avg. LR Enc. Train. Loss": f"{lr_encoder_train_loss:.4f}",
                    })

            # Logging the training progress bar
            logging.info(pre_train_lr_encoder_pbar)

            # Adjust learning rate
            pre_train_lr_encoder_scheduler.step()

            # Increment epoch count
            pre_train_lr_encoder_dataset.increment_epoch()

            # Release unused memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else (torch.mps.empty_cache() if torch.backends.mps.is_available() else None)

        # Replace forward method to extract features
        lr_encoder.forward = lr_encoder.extract_features

    # Training and validation loop
    print_divisor()

    for epoch in range(start_epoch, end_epoch):
        # Set the noise predictor to train mode
        noise_predictor.train()
        noise_predictor_train_loss = 0.0

        # Set the LR encoder to train or evaluation mode if applicable
        if config["MODEL"]["LR_CONDITION_ENCODING"]:
            lr_encoder.train() if not config["TRAINING"]["FIX_LR_ENCODER"] else lr_encoder.eval()

        # Define training progress bar
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{end_epoch}")

        for batch_index, (lr_volumes, hr_volumes, _) in enumerate(train_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)
            hr_volumes = hr_volumes.to(device)

            # Compute residual volumes
            if config["MODEL"]["RESIDUAL_PREDICTION"]:
                # Upsample volumes
                lr_volumes_up = torch.stack([upsample(lr_volume, config["MODEL"]["SCALE_FACTOR"]) for lr_volume in lr_volumes])

                volumes_0 = hr_volumes - lr_volumes_up
            else:
                volumes_0 = hr_volumes

            # Sample timesteps
            timesteps = gaussian_diffusion.sample_timesteps(len(volumes_0)).to(device)

            # Sample timestep volumes
            volumes_t, eps_t_real = gaussian_diffusion.forward_diffusion(volumes_0, timesteps)

            # Zero the gradients
            noise_predictor_optimizer.zero_grad()

            # Forward pass
            eps_t_pred = noise_predictor(volumes_t, (lr_encoder(lr_volumes) if config["MODEL"]["LR_CONDITION_ENCODING"] else lr_volumes), timesteps)

            # Compute diffusion loss
            diffusion_loss = diffusion_criterion(eps_t_pred, eps_t_real)

            # Noise Predictor Total loss
            noise_predictor_total_loss = config['LOSS_WEIGHTS']['NOISE_PREDICTOR']['DIFFUSION_LOSS'] * diffusion_loss

            # Backward pass
            noise_predictor_total_loss.backward()
            noise_predictor_optimizer.step()

            # Adjust learning rate
            noise_predictor_scheduler.step()

            # Update the training loss
            noise_predictor_train_loss += noise_predictor_total_loss.item()
            train_pbar.set_postfix({
                'Noise Pred. Train. Loss': f'{noise_predictor_total_loss.item():.4f}',
            })

            # Compute average training loss for this epoch
            if batch_index + 1 == len(train_pbar):
                noise_predictor_train_loss /= len(train_dataloader)
                noise_predictor_train_losses[epoch + 1] = noise_predictor_train_loss
                train_pbar.set_postfix({
                    "Avg. Noise Pred. Train. Loss": f"{noise_predictor_train_loss:.4f}"
                })

        # Logging the training progress bar
        logging.info(train_pbar)

        # Release unused memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else (torch.mps.empty_cache() if torch.backends.mps.is_available() else None)

        if args.val:
            # Set the noise predictor to evaluation mode
            noise_predictor.eval()
            noise_predictor_val_loss = 0.0

            # Set the LR encoder to evaluation mode if applicable
            if config["MODEL"]["LR_CONDITION_ENCODING"]:
                lr_encoder.eval()

            # Define validation progress bar
            val_pbar = tqdm(val_dataloader, desc="Validating")

            with torch.no_grad():
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(val_pbar):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)
                    hr_volumes = hr_volumes.to(device)

                    # Compute residual volumes
                    if config["MODEL"]["RESIDUAL_PREDICTION"]:
                        # Upsample volumes
                        lr_volumes_up = torch.stack([upsample(lr_volume, config["MODEL"]["SCALE_FACTOR"]) for lr_volume in lr_volumes])

                        volumes_0 = hr_volumes - lr_volumes_up
                    else:
                        volumes_0 = hr_volumes

                    # Sample timesteps
                    timesteps = gaussian_diffusion.sample_timesteps(len(volumes_0)).to(device)

                    # Sample timestep volumes
                    volumes_t, eps_t_real = gaussian_diffusion.forward_diffusion(volumes_0, timesteps)

                    # Forward pass
                    eps_t_pred = noise_predictor_inferer(inputs=volumes_t, conditions=lr_volumes, timesteps=timesteps, inputs_network=noise_predictor, conditions_network=(lr_encoder if config["MODEL"]["LR_CONDITION_ENCODING"] else None))

                    # Compute diffusion loss
                    diffusion_loss = diffusion_criterion(eps_t_pred, eps_t_real)

                    # Noise Predictor Total loss
                    noise_predictor_total_loss = config['LOSS_WEIGHTS']['NOISE_PREDICTOR']['DIFFUSION_LOSS'] * diffusion_loss

                    # Update the validation loss
                    noise_predictor_val_loss += noise_predictor_total_loss.item()
                    val_pbar.set_postfix({
                        "Noise Pred. Val. Loss": f"{noise_predictor_total_loss.item():.4f}"
                    })

                    # Compute average validation loss for this epoch
                    if batch_index + 1 == len(val_pbar):
                        noise_predictor_val_loss /= len(val_dataloader)
                        noise_predictor_val_losses[epoch + 1] = noise_predictor_val_loss
                        val_pbar.set_postfix({
                            "Avg. Noise Pred. Val. Loss": f"{noise_predictor_val_loss:.4f}"
                        })

                # Logging the validation progress bar
                logging.info(val_pbar)

                # Release unused memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else (torch.mps.empty_cache() if torch.backends.mps.is_available() else None)

        # Increment epoch count
        train_dataset.increment_epoch()
        val_dataset.increment_epoch() if args.val else None

        # Plot reference sample figure
        if epoch == 0:
            save_sample(
                lr_volume=clamp(lr_volumes[-1], _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std),
                gen_volume=clamp(clamp(denormalize(torch.randn_like(hr_volumes[-1]), mean=mean, std=std), _min=(_min if args.clamp else None), _max=(_max if args.clamp else None), mean=mean, std=std), _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std),
                gt_volume=clamp(hr_volumes[-1], _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std),
                bins=100,
                _min=(args.sample_pixel_range[0] if args.sample_pixel_range[0] is not None else _min), _max=(args.sample_pixel_range[1] if args.sample_pixel_range[1] is not None else _max), mean=mean, std=std,
                save_filename=os.path.join(experiment_path, 'figures', 'train', 'samples', 'reference'),
                fig_size=(3, 3)
            ) if args.val else None

        # Save checkpoint
        if (epoch + 1) % config["TRAINING"]["SAVE_INTERVAL"] == 0:
            # Assemble checkpoint data
            checkpoint_data = {
                'noise_predictor_state_dict': noise_predictor.state_dict(),
                'noise_predictor_optimizer_state_dict': noise_predictor_optimizer.state_dict(),
                'noise_predictor_scheduler_state_dict': noise_predictor_scheduler.state_dict(),
                'gaussian_diffusion_num_timesteps': config["MODEL"]["NUM_TIMESTEPS"],
                'gaussian_diffusion_beta_schedule': config["MODEL"]["BETA_SCHEDULE"],
                'gaussian_beta_schedule_parameters': config["BETA_SCHEDULE_PARAMETERS"][config["MODEL"]["BETA_SCHEDULE"]],
                'gaussian_diffusion_residual_prediction': config["MODEL"]["RESIDUAL_PREDICTION"],
                'gaussian_diffusion_lr_condition_encoding': config["MODEL"]["LR_CONDITION_ENCODING"],
                'gaussian_diffusion_scale_factor': config["MODEL"]["SCALE_FACTOR"],
                'noise_predictor_in_channels': config["MODEL"]["NOISE_PREDICTOR"]["IN_CHANNELS"],
                'noise_predictor_out_channels': config["MODEL"]["NOISE_PREDICTOR"]["OUT_CHANNELS"],
                'noise_predictor_num_inner_features': config["MODEL"]["NOISE_PREDICTOR"]["NUM_INNER_FEATURES"],
                'noise_predictor_feature_multipliers': config["MODEL"]["NOISE_PREDICTOR"]["FEATURE_MULTIPLIERS"],
                'noise_predictor_num_rdb_blocks': config["MODEL"]["NOISE_PREDICTOR"]["NUM_RDB_BLOCKS"],
                'noise_predictor_num_rdb_conv_blocks': config["MODEL"]["NOISE_PREDICTOR"]["NUM_RDB_CONV_BLOCKS"],
                'noise_predictor_num_rdb_growth_features': config["MODEL"]["NOISE_PREDICTOR"]["NUM_RDB_GROWTH_FEATURES"],
                'noise_predictor_attention_levels': config["MODEL"]["NOISE_PREDICTOR"]["ATTENTION_LEVELS"],
                'noise_predictor_num_attention_heads': config["MODEL"]["NOISE_PREDICTOR"]["NUM_ATTENTION_HEADS"],
                'noise_predictor_num_groups': config["MODEL"]["NOISE_PREDICTOR"]["NUM_GROUPS"],
                'noise_predictor_train_losses': noise_predictor_train_losses,
                'noise_predictor_val_losses': noise_predictor_val_losses,
                'epoch': epoch,
                'training_lr_patch_size': train_dataset.get_low_resolution_shape(only_spatial_dims=True),
                'training_data_min': _min,
                'training_data_max': _max,
                'training_data_mean': mean,
                'training_data_std': std
            }

            if config["MODEL"]["LR_CONDITION_ENCODING"]:
                checkpoint_data['lr_encoder_state_dict'] = lr_encoder.state_dict()
                checkpoint_data['lr_encoder_in_channels'] = config["MODEL"]["LR_ENCODER"]["IN_CHANNELS"]
                checkpoint_data['lr_encoder_out_channels'] = config["MODEL"]["LR_ENCODER"]["OUT_CHANNELS"]
                checkpoint_data['lr_encoder_num_features'] = config["MODEL"]["LR_ENCODER"]["NUM_FEATURES"]
                checkpoint_data['lr_encoder_num_rrdb_blocks'] = config["MODEL"]["LR_ENCODER"]["NUM_RRDB_BLOCKS"]
                checkpoint_data['lr_encoder_num_rrdb_db_blocks'] = config["MODEL"]["LR_ENCODER"]["NUM_RRDB_DB_BLOCKS"]
                checkpoint_data['lr_encoder_num_rrdb_db_conv_blocks'] = config["MODEL"]["LR_ENCODER"]["NUM_RRDB_DB_CONV_BLOCKS"]
                checkpoint_data['lr_encoder_num_rrdb_db_growth_features'] = config["MODEL"]["LR_ENCODER"]["NUM_RRDB_DB_GROWTH_FEATURES"]

            torch.save(checkpoint_data, os.path.join(experiment_path, 'checkpoints', f'{epoch + 1}.pth'))

            # Plot sample figure
            save_sample(
                lr_volume=clamp(lr_volumes[-1], _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std),
                gen_volume=clamp(clamp(gaussian_diffusion.sample(lr_volumes[-1].unsqueeze(dim=0), noise_predictor, (lr_encoder if config["MODEL"]["LR_CONDITION_ENCODING"] else None), noise_predictor_inferer).squeeze(dim=0), _min=(_min if args.clamp else None), _max=(_max if args.clamp else None), mean=mean, std=std), _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std),
                gt_volume=clamp(hr_volumes[-1], _min=args.sample_pixel_range[0], _max=args.sample_pixel_range[1], mean=mean, std=std),
                bins=100,
                _min=(args.sample_pixel_range[0] if args.sample_pixel_range[0] is not None else _min), _max=(args.sample_pixel_range[1] if args.sample_pixel_range[1] is not None else _max), mean=mean, std=std,
                save_filename=os.path.join(experiment_path, 'figures', 'train', 'samples', f'{epoch + 1}'),
                fig_size=(3, 3)
            ) if args.val else None

            # Plot noise predictor loss
            draw_line_chart(
                data=[
                    {'Epoch': noise_predictor_train_losses.keys(), 'Loss': noise_predictor_train_losses.values(), 'Label': 'Train Loss'},
                    {'Epoch': noise_predictor_val_losses.keys(), 'Loss': noise_predictor_val_losses.values(), 'Label': 'Validation Loss'},
                ],
                title='Noise Predictor Training and Validation Loss',
                x_label='Epoch',
                y_label='Loss',
                save_filename=os.path.join(experiment_path, 'figures', 'train', 'noise_predictor_loss')
            )

        # Release unused memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else (torch.mps.empty_cache() if torch.backends.mps.is_available() else None)

    print_divisor()


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--val', default=False, help='Apply validation during training', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--resume', default=None, help="Path to experiment directory to be resumed", type=str)
    parser.add_argument('--clamp', default=True, help='Clamp generated data to original data range', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--sample-pixel-range', default=(None, None), help='Window range of pixel values (min, max). For multi-channel scaling provide ((first_channel_min, ..., last_channel_min), (first_channel_max, ..., last_channel_max))', type=lambda arg: np.array(ast.literal_eval(arg)))
    args = parser.parse_args()

    if args.resume:
        experiment_path = args.resume
        config_path = os.path.join(args.resume, "configs", "config.yaml")
        checkpoint_path = max(glob(os.path.join(args.resume, "checkpoints", "*.pth")), key=os.path.getctime)
    else:
        config_path = os.path.join('..', '..', 'configs', 'config.yaml')

    # Load configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Build experiment directory
    if not args.resume:
        experiment_path = os.path.join('..', '..', 'experiments', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

        # Create experiment folder
        create_experiment_directory(experiment_path=experiment_path, config_path=config_path)

    # Set up logging
    setup_logging(os.path.join(experiment_path, "logs", "train", f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'))

    main()

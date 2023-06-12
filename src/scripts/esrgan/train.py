import os
import sys
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
from torchview import draw_graph
from tqdm import tqdm
from glob import glob
sys.path.append(os.path.join('..', '..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator, Discriminator
from src.utils.losses import VGGLoss
from src.utils.utils import create_experiment_directory, compute_data_statistics, draw_line_chart
from src.utils.logger import setup_logger


def main():
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Set seed for reproducibility
    torch.manual_seed(20)

    # Compute training data statistics
    _min, _max, mean, std = compute_data_statistics(data_dir=config["DATA"]["TRAIN_DIR_PATH"])

    # Define datasets and data loaders
    train_dataset = SuperResolutionDataset(data_dir=config["DATA"]["TRAIN_DIR_PATH"], scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"], _min=_min, _max=_max, mean=mean, std=std, augmentation=config["TRAINING"]["AUGMENTATION"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate)
    val_dataset = SuperResolutionDataset(data_dir=config["DATA"]["VAL_DIR_PATH"], scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"], _min=_min, _max=_max, mean=mean, std=std, augmentation=False) if args.val else None
    val_dataloader = DataLoader(val_dataset, batch_size=config["VALIDATION"]["BATCH_SIZE"], num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate) if args.val else None

    # Define model
    generator = Generator(
        in_channels=config["MODEL"]["GENERATOR"]["IN_CHANNEL"],
        out_channels=config["MODEL"]["GENERATOR"]["OUT_CHANNEL"],
        num_features=config["MODEL"]["GENERATOR"]["NUM_FEATURES"],
        num_growth_features=config["MODEL"]["GENERATOR"]["NUM_GROWTH_FEATURES"],
        num_rrdb_blocks=config["MODEL"]["GENERATOR"]["NUM_RRDB_BLOCKS"],
        num_rdb_blocks=config["MODEL"]["GENERATOR"]["NUM_RDB_BLOCKS"],
        num_rdb_conv_blocks=config["MODEL"]["GENERATOR"]["NUM_RDB_CONV_BLOCKS"],
        scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"],
    ).to(device)

    discriminator = Discriminator(
        in_channels=config["MODEL"]["DISCRIMINATOR"]["IN_CHANNEL"],
        num_features=config["MODEL"]["DISCRIMINATOR"]["NUM_FEATURES"],
        num_conv_blocks=config["MODEL"]["DISCRIMINATOR"]["NUM_CONV_BLOCKS"],
        conv_blocks_output_size=config["MODEL"]["DISCRIMINATOR"]["CONV_BLOCKS_OUTPUT_SIZE"],
    ).to(device)

    # Log Generator and Discriminator
    summary(generator, input_size=(1, *train_dataset.get_low_resolution_shape()), depth=1, device=device)
    draw_graph(generator, input_size=(1, *train_dataset.get_low_resolution_shape()), depth=3, expand_nested=True, filename='generator_architecture', directory=os.path.join(experiment_path, 'figures', 'train'), device=device).visual_graph.view()

    summary(discriminator, input_size=(1, *train_dataset.get_high_resolution_shape()), depth=1, device=device)
    draw_graph(discriminator, input_size=(1, *train_dataset.get_high_resolution_shape()), depth=3, expand_nested=True, filename='discriminator_architecture', directory=os.path.join(experiment_path, 'figures', 'train'), device=device).visual_graph.view()

    # Define optimizers
    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['OPTIMIZER']['GENERATOR']['LEARNING_RATE'],
        betas=(config['OPTIMIZER']['GENERATOR']['BETA_1'], config['OPTIMIZER']['GENERATOR']['BETA_2']),
        weight_decay=config['OPTIMIZER']['GENERATOR']['WEIGHT_DECAY']
    )

    discriminator_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config['OPTIMIZER']['DISCRIMINATOR']['LEARNING_RATE'],
        betas=(config['OPTIMIZER']['DISCRIMINATOR']['BETA_1'], config['OPTIMIZER']['DISCRIMINATOR']['BETA_2']),
        weight_decay=config['OPTIMIZER']['DISCRIMINATOR']['WEIGHT_DECAY']
    )

    # Define the loss function
    perceptual_criterion = VGGLoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion = nn.L1Loss().to(device)

    # Load pre-trained model
    if config["MODEL"]["GENERATOR"]["PRE_TRAINED_PATH"]:
        # Load the pre-trained checkpoint
        checkpoint = torch.load(config["MODEL"]["GENERATOR"]["PRE_TRAINED_PATH"])

        # Initialize generator weights
        generator.load_state_dict(checkpoint['generator_state_dict'])
    elif config["MODEL"]["DISCRIMINATOR"]["PRE_TRAINED_PATH"]:
        # Load the pre-trained checkpoint
        checkpoint = torch.load(config["MODEL"]["DISCRIMINATOR"]["PRE_TRAINED_PATH"])

        # Initialize discriminator weights
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    elif config['MODEL']['PRE_TRAINED_PATH']:
        # Load the pre-trained checkpoint
        checkpoint = torch.load(config["MODEL"]["PRE_TRAINED_PATH"])

        # Initialize generator and discriminator weights
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Define number of epochs and resume training if applicable
    if args.resume:
        # Load the saved checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Initialize weights, optimizers and epoch count
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_dataset.update_epoch(start_epoch)
        val_dataset.update_epoch(start_epoch) if args.val else None
        generator_train_losses = checkpoint['generator_train_losses']
        generator_val_losses = checkpoint['generator_val_losses']
        discriminator_train_losses = checkpoint['discriminator_train_losses']
        discriminator_val_losses = checkpoint['discriminator_val_losses']
    else:
        start_epoch = 0
        generator_train_losses = dict()
        generator_val_losses = dict()
        discriminator_train_losses = dict()
        discriminator_val_losses = dict()

    end_epoch = config['TRAINING']['EPOCHS']

    # Training and validation loop
    for epoch in range(start_epoch, end_epoch):
        # Set the model to train mode
        generator.train()
        discriminator.train()
        generator_train_loss = 0.0
        discriminator_train_loss = 0.0

        # Define training progress bar
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{end_epoch}")

        for batch_index, (lr_volumes, hr_volumes, _) in enumerate(train_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)
            hr_volumes = hr_volumes.to(device)

            # Define adversarial ground truths
            real_labels_true = torch.tensor(np.ones((hr_volumes.size(0), *discriminator.output_shape)), requires_grad=False).to(device)
            fake_labels_true = torch.tensor(np.zeros((lr_volumes.size(0), *discriminator.output_shape)), requires_grad=False).to(device)

            #=========================#
            #     Train Generator     #
            #=========================#

            # Zero the gradients
            generator_optimizer.zero_grad()

            # Forward pass
            gen_volumes = generator(lr_volumes)

            real_labels_pred = discriminator(hr_volumes).detach()
            fake_labels_pred = discriminator(gen_volumes)

            # Compute perceptual loss
            perceptual_loss = perceptual_criterion(gen_volumes, hr_volumes)

            # Compute adversarial loss (relativistic average GAN)
            adversarial_loss = adversarial_criterion(fake_labels_pred - real_labels_pred.mean(0, keepdim=True), real_labels_true)

            # Compute pixel-wise loss
            content_loss = content_criterion(gen_volumes, hr_volumes)

            # Generator Total loss
            generator_total_loss = perceptual_loss + config['LOSS_WEIGHTS']['GENERATOR']['ADVERSARIAL_LOSS'] * adversarial_loss + config['LOSS_WEIGHTS']['GENERATOR']['CONTENT_LOSS'] * content_loss

            # Backward pass
            generator_total_loss.backward()
            generator_optimizer.step()

            #=========================#
            #   Train Discriminator   #
            #=========================#

            # Zero the gradients
            discriminator_optimizer.zero_grad()

            # Forward pass
            real_labels_pred = discriminator(hr_volumes)
            fake_labels_pred = discriminator(gen_volumes.detach())

            # Compute adversarial loss (relativistic average GAN)
            adversarial_loss_real = adversarial_criterion(real_labels_pred - fake_labels_pred.mean(0, keepdim=True), real_labels_true)
            adversarial_loss_fake = adversarial_criterion(fake_labels_pred - real_labels_pred.mean(0, keepdim=True), fake_labels_true)

            # Discriminator Total loss
            discriminator_total_loss = (adversarial_loss_real + adversarial_loss_fake) / 2

            # Backward pass
            discriminator_total_loss.backward()
            discriminator_optimizer.step()

            # Update the training loss
            generator_train_loss += generator_total_loss.item()
            discriminator_train_loss += discriminator_total_loss.item()
            train_pbar.set_postfix({
                'Gen. Train. Loss': f'{generator_total_loss.item():.4f}',
                'Disc. Train. Loss': f'{discriminator_total_loss.item():.4f}'
            })

            # Compute average training loss for this epoch
            if batch_index + 1 == len(train_pbar):
                generator_train_loss /= len(train_dataloader)
                generator_train_losses[epoch + 1] = generator_train_loss
                discriminator_train_loss /= len(train_dataloader)
                discriminator_train_losses[epoch + 1] = discriminator_train_loss
                train_pbar.set_postfix({
                    "Avg. Gen. Train. Loss": f"{generator_train_loss:.4f}",
                    "Avg. Disc. Train. Loss": f"{discriminator_train_loss:.4f}"
                })

        # Logging the training progress bar
        logging.info(train_pbar)

        if args.val:
            # Set the model to evaluation mode
            generator.eval()
            discriminator.eval()
            generator_val_loss = 0.0
            discriminator_val_loss = 0.0

            # Define validation progress bar
            val_pbar = tqdm(val_dataloader, desc="Validating")

            with torch.no_grad():
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(val_pbar):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)
                    hr_volumes = hr_volumes.to(device)

                    # Define adversarial ground truths
                    real_labels_true = torch.tensor(np.ones((hr_volumes.size(0), *discriminator.output_shape))).to(device)
                    fake_labels_true = torch.tensor(np.zeros((lr_volumes.size(0), *discriminator.output_shape))).to(device)

                    #=========================#
                    #    Validate Generator   #
                    #=========================#

                    # Forward pass
                    gen_volumes = generator(lr_volumes)

                    real_labels_pred = discriminator(hr_volumes)
                    fake_labels_pred = discriminator(gen_volumes)

                    # Compute perceptual loss
                    perceptual_loss = perceptual_criterion(gen_volumes, hr_volumes)

                    # Compute adversarial loss (relativistic average GAN)
                    adversarial_loss = adversarial_criterion(fake_labels_pred - real_labels_pred.mean(0, keepdim=True), real_labels_true)

                    # Compute pixel-wise loss
                    content_loss = content_criterion(gen_volumes, hr_volumes)

                    # Generator Total loss
                    generator_total_loss = perceptual_loss + config['LOSS_WEIGHTS']['GENERATOR']['ADVERSARIAL_LOSS'] * adversarial_loss + config['LOSS_WEIGHTS']['GENERATOR']['CONTENT_LOSS'] * content_loss

                    #=========================#
                    #  Validate Discriminator #
                    #=========================#

                    # Forward pass
                    real_labels_pred = discriminator(hr_volumes)
                    fake_labels_pred = discriminator(gen_volumes)

                    # Compute adversarial loss (relativistic average GAN)
                    adversarial_loss_real = adversarial_criterion(real_labels_pred - fake_labels_pred.mean(0, keepdim=True), real_labels_true)
                    adversarial_loss_fake = adversarial_criterion(fake_labels_pred - real_labels_pred.mean(0, keepdim=True), fake_labels_true)

                    # Discriminator Total loss
                    discriminator_total_loss = (adversarial_loss_real + adversarial_loss_fake) / 2

                    # Update the validation loss
                    generator_val_loss += generator_total_loss.item()
                    discriminator_val_loss += discriminator_total_loss.item()
                    val_pbar.set_postfix({
                        "Gen. Val. Loss": f"{generator_total_loss.item():.4f}",
                        "Disc. Val. Loss": f"{discriminator_total_loss.item():.4f}"
                    })

                    # Compute average validation loss for this epoch
                    if batch_index + 1 == len(val_pbar):
                        generator_val_loss /= len(val_dataloader)
                        generator_val_losses[epoch + 1] = generator_val_loss
                        discriminator_val_loss /= len(val_dataloader)
                        discriminator_val_losses[epoch + 1] = discriminator_val_loss
                        val_pbar.set_postfix({
                            "Gen. Avg. Val. Loss": f"{generator_val_loss:.4f}",
                            "Disc. Avg. Val. Loss": f"{discriminator_val_loss:.4f}"
                        })

            # Logging the validation progress bar
            logging.info(val_pbar)

        # Increment epoch count
        train_dataset.increment_epoch()
        val_dataset.increment_epoch() if args.val else None

        # Save checkpoint
        if (epoch + 1) % config["TRAINING"]["SAVE_INTERVAL"] == 0:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'generator_in_channels': config["MODEL"]["GENERATOR"]["IN_CHANNEL"],
                'generator_out_channels': config["MODEL"]["GENERATOR"]["OUT_CHANNEL"],
                'generator_num_features': config["MODEL"]["GENERATOR"]["NUM_FEATURES"],
                'generator_num_growth_features': config["MODEL"]["GENERATOR"]["NUM_GROWTH_FEATURES"],
                'generator_num_rrdb_blocks': config["MODEL"]["GENERATOR"]["NUM_RRDB_BLOCKS"],
                'generator_num_rdb_blocks': config["MODEL"]["GENERATOR"]["NUM_RDB_BLOCKS"],
                'generator_num_rdb_conv_blocks': config["MODEL"]["GENERATOR"]["NUM_RDB_CONV_BLOCKS"],
                'generator_scale_factor': config["MODEL"]["GENERATOR"]["SCALE_FACTOR"],
                'generator_train_losses': generator_train_losses,
                'generator_val_losses': generator_val_losses,
                'discriminator_train_losses': discriminator_train_losses,
                'discriminator_val_losses': discriminator_val_losses,
                'epoch': epoch,
                'training_data_min': _min,
                'training_data_max': _max,
                'training_data_mean': mean,
                'training_data_std': std
            }, os.path.join(experiment_path, 'checkpoints', f'{epoch + 1}.pth'))

            draw_line_chart(
                data=[
                    {'Epoch': generator_train_losses.keys(), 'Loss': generator_train_losses.values(), 'Label': 'Train Loss'},
                    {'Epoch': generator_val_losses.keys(), 'Loss': generator_val_losses.values(), 'Label': 'Validation Loss'},
                ],
                title='Generator Training and Validation Loss',
                x_label='Epoch',
                y_label='Loss',
                save_dir_path=os.path.join(experiment_path, 'figures', 'train', 'generator_loss')
            )

            draw_line_chart(
                data=[
                    {'Epoch': discriminator_train_losses.keys(), 'Loss': discriminator_train_losses.values(), 'Label': 'Train Loss'},
                    {'Epoch': discriminator_val_losses.keys(), 'Loss': discriminator_val_losses.values(), 'Label': 'Validation Loss'},
                ],
                title='Discriminator Training and Validation Loss',
                x_label='Epoch',
                y_label='Loss',
                save_dir_path=os.path.join(experiment_path, 'figures', 'train', 'discriminator_loss')
            )


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--val', default=True, help='Apply validation during training', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--resume', default=None, help="Path to experiment directory to be resumed", type=str)
    args = parser.parse_args()

    if args.resume:
        experiment_path = args.resume
        config_path = os.path.join(args.resume, "config.yaml")
        checkpoint_path = max(glob(os.path.join(args.resume, "checkpoints", "*.pth")), key=os.path.getctime)
    else:
        experiment_path = os.path.join('..', '..', '..', 'experiments', "esrgan", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        config_path = os.path.join('..', '..', '..', 'configuration', "esrgan", 'config.yaml')

        # Create experiment folder
        create_experiment_directory(experiment_path, config_path)

    # Load configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logger
    setup_logger(os.path.join(experiment_path, "logs", "train", f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'))

    main()

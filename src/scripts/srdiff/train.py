import os
import sys
import argparse
import datetime
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torchview import draw_graph
from tqdm import tqdm
from glob import glob
sys.path.append(os.path.join('..', '..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.srdiff import ????????????????????????????????????????
from src.utils.utils import create_experiment_directory, compute_data_statistics, plot_losses
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
    val_dataset = SuperResolutionDataset(data_dir=config["DATA"]["VAL_DIR_PATH"], scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"], _min=_min, _max=_max, mean=mean, std=std, augmentation=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config["VALIDATION"]["BATCH_SIZE"], shuffle=True, num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate)

    # Define model
    model = SRDiff(??????).to(device)

    # Log model
    summary(model, input_size=(1, 1, 200, 200), depth=1)
    draw_graph(model, input_size=(1, 1, 200, 200), depth=3, expand_nested=True, filename='architecture', directory=os.path.join(experiment_path, 'figures', 'train')).visual_graph.view()

    # Define optimizers
    optimizer = optim.Adam(model.parameters(), lr=config['OPTIMIZER']['LEARNING_RATE'], betas=(config['OPTIMIZER']['BETA_1'], config['OPTIMIZER']['BETA_2']))

    # Define the loss function
    criterion = nn.L1Loss().to(device)

    # Load pre-trained model
    if config["MODEL"]["PRE_TRAINED_PATH"]:
        # Load the pre-trained checkpoint
        checkpoint = torch.load(config["MODEL"]["PRE_TRAINED_PATH"])

        # Initialize generator weights
        model.load_state_dict(checkpoint['model_state_dict'])

    # Define number of epochs and resume training if applicable
    if args.resume:
        # Load the saved checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Initialize weights, optimizers and epoch count
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_dataset.update_epoch(start_epoch)
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
    else:
        start_epoch = 0
        train_losses = dict()
        val_losses = dict()

    end_epoch = config['TRAINING']['EPOCHS']

    # Training and validation loop
    for epoch in range(start_epoch, end_epoch):
        # Set the model to train mode
        model.train()
        train_loss = 0.0

        # Define training progress bar
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{end_epoch}")

        for batch_index, (lr_volumes, hr_volumes, _) in enumerate(train_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)
            hr_volumes = hr_volumes.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            gen_volumes = model(lr_volumes)
            loss = criterion(gen_volumes, hr_volumes)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the training loss
            train_loss += loss.item()
            train_pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

            # Compute average training loss for this epoch
            if batch_index + 1 == len(train_pbar):
                train_loss /= len(train_dataloader)
                train_losses[epoch + 1] = train_loss
                train_pbar.set_postfix({"Avg. Train Loss": f"{train_loss:.4f}"})

        # Logging the training progress bar
        logging.info(train_pbar)

        if args.val:
            # Set the model to evaluation mode
            model.eval()
            val_loss = 0.0

            # Define validation progress bar
            val_pbar = tqdm(val_dataloader, desc="Validating")

            with torch.no_grad():
                for batch_index, (lr_volumes, hr_volumes, _) in enumerate(val_pbar):
                    # Move inputs to the designated device
                    lr_volumes = lr_volumes.to(device)
                    hr_volumes = hr_volumes.to(device)

                    # Forward pass
                    gen_volumes = model(lr_volumes)
                    loss = criterion(gen_volumes, hr_volumes)

                    # Update the validation loss
                    val_loss += loss.item()
                    val_pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

                    # Compute average validation loss for this epoch
                    if batch_index + 1 == len(val_pbar):
                        val_loss /= len(val_dataloader)
                        val_losses[epoch + 1] = val_loss
                        val_pbar.set_postfix({"Avg. Val Loss": f"{val_loss:.4f}"})

            # Logging the validation progress bar
            logging.info(val_pbar)

        # Increment epoch count
        train_dataset.increment_epoch()

        # Save checkpoint
        if (epoch + 1) % config["TRAINING"]["SAVE_INTERVAL"] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(experiment_path, 'checkpoints', f'{epoch + 1}.pth'))

            plot_losses(
                loss_dicts={'Train Loss': train_losses, 'Validation Loss': val_losses},
                title='Training and Validation Loss',
                save_dir_path=os.path.join(experiment_path, 'figures', 'train', 'loss')
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
        experiment_path = os.path.join('..', '..', '..', 'experiments', "srdiff", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        config_path = os.path.join('..', '..', '..', 'configuration', "srdiff", 'config.yaml')

        # Create experiment folder
        create_experiment_directory(experiment_path, config_path)

    # Load configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logger
    setup_logger(os.path.join(experiment_path, "logs", "train", f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'))

    main()

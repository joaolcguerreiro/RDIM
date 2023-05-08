import os
import sys
import torch
import shutil
import argparse
import datetime
import logging
import yaml
from torch.utils.data import DataLoader
from torchinfo import summary
from torchview import draw_graph
from tqdm import tqdm
from glob import glob
sys.path.append(os.path.join('..', '..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator, Discriminator, ESRGAN


def main():
    # Define device
    torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Set seed for reproducibility
    torch.manual_seed(20)

    # Compute training data statistics
    mean, std = SuperResolutionDataset.compute_data_statistics(data_dir=config["DATA"]["TRAIN_DATA_DIR_PATH"], batch_size=config["DATA"]["BATCH_SIZE"])

    # Define datasets and data loaders
    train_dataset = SuperResolutionDataset(data_dir=config["DATA"]["TRAIN_DATA_DIR_PATH"], mean=mean, std=std, scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"], augmentation=config["TRAINING"]["AUGMENTATION"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["DATA"]["BATCH_SIZE"], shuffle=True)
    val_dataset = SuperResolutionDataset(data_dir=config["DATA"]["VAL_DATA_DIR_PATH"], mean=mean, std=std, scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"], augmentation=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config["DATA"]["BATCH_SIZE"], shuffle=True)

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
    )

    discriminator = Discriminator(
        in_channels=config["MODEL"]["DISCRIMINATOR"]["IN_CHANNEL"],
        num_features=config["MODEL"]["DISCRIMINATOR"]["NUM_FEATURES"],
        num_conv_blocks=config["MODEL"]["DISCRIMINATOR"]["NUM_CONV_BLOCKS"],
        conv_blocks_output_size=config["MODEL"]["DISCRIMINATOR"]["CONV_BLOCKS_OUTPUT_SIZE"],
    )

    model = ESRGAN(generator=generator, discriminator=discriminator)

    # Log model
    summary(model, input_size=(1, 1, 200, 200), depth=1)
    draw_graph(model, input_size=(1, 1, 200, 200), depth=3).visual_graph.view()

    # Define optimizers
    generator_optimizer = optim.SGD(gan.generator.parameters(), lr=0.01)
    discriminator_optimizer = optim.SGD(gan.discriminator.parameters(), lr=0.01)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # adversarial_loss = nn.BCEWithLogitsLoss()
    # content_loss = nn.L1Loss()

    # Load pre-trained model
    if config["MODEL"]["GENERATOR"]["PRE_TRAINED_PATH"]:
        # Load the pre-trained checkpoint
        checkpoint = torch.load(config["MODEL"]["GENERATOR"]["PRE_TRAINED_PATH"])

        # Initialize generator weights
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
    elif config["MODEL"]["DISCRIMINATOR"]["PRE_TRAINED_PATH"]:
        # Load the pre-trained checkpoint
        checkpoint = torch.load(config["MODEL"]["DISCRIMINATOR"]["PRE_TRAINED_PATH"])

        # Initialize discriminator weights
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Define number of epochs and resume training if applicable
    if args.resume:
        # Load the saved checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Initialize weights, optimizers and epoch count
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_dataset.update_epoch(start_epoch)
    else:
        start_epoch = 0

    end_epoch = config['TRAINING']['EPOCHS']

    # Training and validation loop
    for epoch in range(start_epoch, end_epoch):
        # Set the model to train mode
        model.train()
        train_loss = 0.0

        # Define training progress bar
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{end_epoch}")

        for batch_index, (lr_volumes, hr_volumes) in enumerate(train_pbar):
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
                train_pbar.set_postfix({"Avg. Train Loss": f"{train_loss:.4f}"})

        if args.val:
            # Set the model to evaluation mode
            model.eval()
            val_loss = 0.0

            # Define validation progress bar
            val_pbar = tqdm(val_dataloader, desc="Validating")

            with torch.no_grad():
                for batch_index, (lr_volumes, hr_volumes) in enumerate(val_pbar):
                    # Forward pass
                    gen_volumes = model(lr_volumes)
                    loss = criterion(gen_volumes, hr_volumes)

                    # Update the validation loss
                    val_loss += loss.item()
                    val_pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

                    # Compute average validation loss for this epoch
                    if batch_index + 1 == len(val_pbar):
                        val_loss /= len(val_dataloader)
                        val_pbar.set_postfix({"Avg. Val Loss": f"{val_loss:.4f}"})

        # Increment epoch count
        train_dataset.increment_epoch()

        # Save checkpoint
        if (epoch + 1) % config["TRAINING"]["SAVE_INTERVAL"] == 0:
            torch.save({
                'generator_state_dict': model.generator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'epoch': epoch,
            }, f"{experiment_path}/checkpoints/{epoch + 1}.pth")


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--val', default=True, help='Apply validation during training', type=(lambda arg: (str(arg).lower() == 'true')))
    parser.add_argument('--resume', default=None, help="Path to experiment directory to be resumed", type=str)
    args = parser.parse_args()

    if args.resume:
        experiment_path = args.resume
        checkpoint_path = max(glob.glob(os.path.join(experiment_path, "checkpoints", "*.pth")), key=os.path.getctime)
    else:
        experiment_path = os.path.join('..', '..', '..', 'experiments', "esrgan", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

        # Create experiment folder
        os.makedirs(experiment_path)

        # Create checkpoints, figures, logs and results folder
        os.makedirs(os.path.join(experiment_path, "checkpoints"))
        os.makedirs(*[os.path.join(experiment_path, path) for path in ["figures", os.path.join("figures", "train"), os.path.join("figures", "test"), os.path.join("figures", "inference")]])
        os.makedirs(*[os.path.join(experiment_path, path) for path in ["logs", os.path.join("logs", "train"), os.path.join("logs", "test"), os.path.join("logs", "inference")]])
        os.makedirs(os.path.join(experiment_path, "results"))

        # Clone global config file for reproducibility
        shutil.copyfile(os.path.join('..', '..', '..', '..', 'configuration', 'esrgan', 'config.yaml'), experiment_path)

    # Load configuration file
    with open(os.path.join(experiment_path, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),  # output to console
            logging.FileHandler(os.path.join(experiment_path, "logs", "train", datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))  # output to file
        ])

    # Redirect stdout and stderr to the logging file
    sys.stdout = logging.getLogger().handlers[0].stream
    sys.stderr = logging.getLogger().handlers[0].stream

    main()

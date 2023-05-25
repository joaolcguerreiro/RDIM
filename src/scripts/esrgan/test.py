import os
import sys
import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
sys.path.append(os.path.join('..', '..', '..'))
from src.datasets.sr_dataset import SuperResolutionDataset
from src.models.esrgan import Generator, Discriminator
from src.utils.metrics import SuperResolutionMetrics
from src.utils.utils import plot_metric
from src.utils.logger import setup_logger


def main():
    # Define device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Set seed for reproducibility
    torch.manual_seed(20)

    # Compute training data statistics
    mean, std = SuperResolutionDataset.compute_data_statistics(data_dir=config["DATA"]["TRAIN_DIR_PATH"], batch_size=config["TRAINING"]["BATCH_SIZE"])

    test_dataset = SuperResolutionDataset(data_dir=config["DATA"]["TEST_DIR_PATH"], scale_factor=config["MODEL"]["GENERATOR"]["SCALE_FACTOR"], mean=mean, std=std, augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["TESTING"]["BATCH_SIZE"], shuffle=True, num_workers=config["DATA"]["NUM_WORKERS"])

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
    summary(generator, input_size=(1, *test_dataset.get_low_resolution_shape()), depth=1, device=device)
    summary(discriminator, input_size=(1, *test_dataset.get_high_resolution_shape()), depth=1, device=device)

    # Define metrics
    metrics = SuperResolutionMetrics()

    # Load checkpoint and initialize generator and discriminator weights
    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Set the model to evaluation mode
    generator.eval()
    discriminator.eval()

    # Define validation progress bar
    test_pbar = tqdm(test_dataloader, desc="Testing")

    with torch.no_grad():
        for batch_index, (lr_volumes, hr_volumes) in enumerate(test_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)
            hr_volumes = hr_volumes.to(device)

            # Forward pass
            gen_volumes = generator(lr_volumes)

            # Compute metrics
            scores = metrics.compute_scores(gen_volumes, hr_volumes)
            test_pbar.set_postfix(scores)

        # Logging the validation progress bar
        logging.info(test_pbar)

    # Log metrics
    metrics.summary()

    # Plot metrics
    for metric_name, metric_scores in metrics.get().items():
        plot_metric(
            data={
                'model': ['ESRGAN'] * len(metric_scores),
                'Scale factor': [config["MODEL"]["GENERATOR"]["SCALE_FACTOR"]] * len(metric_scores),
                f'{metric_name}': metric_scores,
            },
            title='Quantitative Results',
            x_label='Scale factor',
            y_label=metric_name,
            hue='model',
            save_dir_path=os.path.join(experiment_path, 'figures', 'test', f'{args.epoch}_metrics')
        )


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Testing Script')
    parser.add_argument('--exp', required=True, help="Path to experiment directory", type=str)
    parser.add_argument('--epoch', required=True, help="Epoch to be tested", type=str)
    args = parser.parse_args()

    # Define experiment folder
    experiment_path = args.exp
    config_path = os.path.join(args.exp, "config.yaml")
    checkpoint_path = os.path.join(args.exp, "checkpoints", f"{args.epoch}.pth")

    # Load configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up logger
    setup_logger(os.path.join(experiment_path, "logs", "test", f'{args.epoch}.log'))

    main()

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
from src.models.esrgan import Generator
from src.utils.metrics import SuperResolutionMetrics
from src.utils.utils import draw_box_plot, save_batch
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

    test_dataset = SuperResolutionDataset(data_dir=config["DATA"]["TEST_DIR_PATH"], scale_factor=checkpoint['generator_scale_factor'], _min=_min, _max=_max, mean=mean, std=std, augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["TESTING"]["BATCH_SIZE"], num_workers=config["DATA"]["NUM_WORKERS"], collate_fn=SuperResolutionDataset.sr_dataset_collate)

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

    # Log Generator and Discriminator
    summary(generator, input_size=(1, *test_dataset.get_low_resolution_shape()), depth=1, device=device)

    # Define metrics
    metrics = SuperResolutionMetrics()

    # Initialize generator and discriminator weights
    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Set the model to evaluation mode
    generator.eval()

    # Define validation progress bar
    test_pbar = tqdm(test_dataloader, desc="Testing")

    with torch.no_grad():
        for batch_index, (lr_volumes, hr_volumes, hr_metadata) in enumerate(test_pbar):
            # Move inputs to the designated device
            lr_volumes = lr_volumes.to(device)
            hr_volumes = hr_volumes.to(device)

            # Forward pass
            gen_volumes = generator(lr_volumes)

            # Compute metrics
            scores = metrics.compute_scores(gen_volumes, hr_volumes)
            test_pbar.set_postfix(scores)

            # Save volumes
            save_batch(gen_volumes, hr_metadata, batch_index, os.path.join(experiment_path, 'results'), mean=mean, std=std)

        # Logging the validation progress bar
        logging.info(test_pbar)

    # Log metrics
    metrics.summary()

    # Plot metrics
    for metric_name, metric_scores in metrics.get().items():
        draw_box_plot(
            data=[
                {'Model': ['ESRGAN'] * len(metric_scores), 'Scale factor': [checkpoint['generator_scale_factor']] * len(metric_scores), metric_name: metric_scores}
            ],
            title='Quantitative Results',
            x_label='Scale factor',
            y_label=metric_name,
            hue='Model',
            save_dir_path=os.path.join(experiment_path, 'figures', 'test', f'{metric_name}', f'{args.epoch}')
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

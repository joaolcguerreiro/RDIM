import os
import torch
import yaml
from torch.utils.data import DataLoader
from src.datasets.sr_dataset import SuperResolutionDataset
from torchinfo import summary
from src.models.esrgan import Generator, Discriminator, ESRGAN


def main():
    # Define optimizers
    optimizer = optim.SGD(gan.generator.parameters(), lr=0.01)

    # Define the learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / config["WARMUP_STEPS"], 1.0)) ### CHECK IF THIS IS RIGHT!! HOW TO DO WARMUP?? IT IS OKAY TO DO WITH GRADIENT ACCUMULATION? HOW TO DO IT?

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Training and validation loop
    for tqdm(epoch in range(100)):
        train_loss = 0.0
        val_loss = 0.0
        correct = 0
        total = 0

        # Training
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        for batch_index, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            (loss / config["GRADIENT_ACCUMULATION_STEPS"]).backward()

            # Update the weights after accumulating gradients for the desired number of steps
            if ((batch_index + 1) % config["GRADIENT_ACCUMULATION_STEPS"] == 0) or (batch_index + 1 == len(train_loader)):
                optimizer.step()

                # Zero the gradients
                optimizer.zero_grad()

                # Update the learning rate for the current step
                scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        print('Epoch: {}/{} | Train Loss: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.2f}%'.format(epoch + 1, 100, train_loss, val_loss, val_acc))


if __name__ == '__main__':
    # Load configuration file
    with open(os.path.join(args.exp, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    # Parse config.yml file relative paths and adapt them to work in the current directory
    parsePath: lambda path: os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))

    # Load model
    generator = Generator(
        in_channels=config["ESRGAN"]["GENERATOR"]["IN_CHANNEL"],
        out_channels=config["ESRGAN"]["GENERATOR"]["OUT_CHANNEL"],
        num_features=config["ESRGAN"]["GENERATOR"]["NUM_FEATURES"],
        num_growth_features=config["ESRGAN"]["GENERATOR"]["NUM_GROWTH_FEATURES"],
        num_rrdb_blocks=config["ESRGAN"]["GENERATOR"]["NUM_RRDB_BLOCKS"],
        num_rdb_blocks=config["ESRGAN"]["GENERATOR"]["NUM_RDB_BLOCKS"],
        num_rdb_conv_blocks=config["ESRGAN"]["GENERATOR"]["NUM_RDB_CONV_BLOCKS"],
        scale_factor=config["ESRGAN"]["GENERATOR"]["SCALE_FACTOR"],
    )

    discriminator = Discriminator(
        in_channels=config["ESRGAN"]["DISCRIMINATOR"]["IN_CHANNEL"],
        num_features=config["ESRGAN"]["DISCRIMINATOR"]["NUM_FEATURES"],
        num_conv_blocks=config["ESRGAN"]["DISCRIMINATOR"]["NUM_CONV_BLOCKS"],
        conv_blocks_output_size=config["ESRGAN"]["DISCRIMINATOR"]["CONV_BLOCKS_OUTPUT_SIZE"],
    )

    model = ESRGAN(generator=generator, discriminator=discriminator)

    # Log model
    summary(model, input_size=(1, 1, 240, 240), depth=1)

    # Load training data
    train_dataset = SuperResolutionDataset(data_dir=parsePath(config["TRAIN"]["DATA_DIR_PATH"]), scale_factor=TRAIN.SCALE_FACTOR, mean=config["TRAIN"]["MEAN"], std=config["TRAIN"]["STD"], augmentation=TRAIN.AUGMENTATION)
    train_dataloader = DataLoader(train_dataset, batch_size=config["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=SuperResolutionDataset.sr_dataset_collate)

    # Load validation data
    val_dataset = SuperResolutionDataset(data_dir=parsePath(config["VAL"]["DATA_DIR_PATH"]), scale_factor=VALIDATION.SCALE_FACTOR, mean=config["TRAIN"]["MEAN"], std=config["TRAIN"]["STD"], augmentation=VALIDATION.AUGMENTATION)
    val_dataloader = DataLoader(train_dataset, batch_size=config["VAL"]["BATCH_SIZE"], shuffle=True, collate_fn=SuperResolutionDataset.sr_dataset_collate)

    main()
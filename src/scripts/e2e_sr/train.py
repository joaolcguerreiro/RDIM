import torch
from torch.utils.data import DataLoader
from src.datasets.sr_brats_dataset import SuperResolutionBraTSDataset


def main():
    # net = Discriminator_VGG_patch_192(3, 64)
    # data = torch.randn(8, 3, 192, 192)
    # output = net(data)
    # print(output.shape)
    pass


if __name__ == '__main__':
    # Load configuration file

    # Load model

    # Load training data
    train_dataset = SuperResolutionBraTSDataset(data_dir=TRAIN.DATA_DIR_PATH, scale_factor=TRAIN.SCALE_FACTOR, augmentation=TRAIN.AUGMENTATION)
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN.BATCH_SIZE, shuffle=True)

    # Load validation data
    val_dataset = SuperResolutionBraTSDataset(data_dir=VALIDATION.DATA_DIR_PATH, scale_factor=VALIDATION.SCALE_FACTOR, augmentation=VALIDATION.AUGMENTATION)
    val_dataloader = DataLoader(train_dataset, batch_size=VALIDATION.BATCH_SIZE, shuffle=True)

    main()

import os
import sys
import logging
import monai.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
sys.path.append(os.path.join('..', '..'))
from src.utils.utils import get_volume_data
from src.utils.transforms import RandGaussianSmooth, RandForegroundGaussianNoise, RandForegroundStreakNoise


class SuperResolutionDataset(Dataset):
    def __init__(self, data_dir, scale_factor, _min=None, _max=None, mean=None, std=None, augmentation=False):
        super().__init__()

        # Data Directory Path
        self.data_dir = data_dir

        # Files
        self.files = os.listdir(data_dir)

        # Track Epoch Count
        self.epoch = 0

        # Scale Factor
        self.scale_factor = scale_factor

        # Common Transforms
        self.initial_transforms = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim='no_channel'),
            transforms.ThresholdIntensity(threshold=(_min - 1), above=True, cval=_min) if _min else transforms.Identity(),
            transforms.ThresholdIntensity(threshold=(_max + 1), above=False, cval=_max) if _max else transforms.Identity(),
        ])

        self.final_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.NormalizeIntensity(subtrahend=mean, divisor=std) if (mean and std) else transforms.Identity(),
        ])

        # Data Augmentation Transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.RandFlip(prob=0.5),
            transforms.RandRotate(prob=0.5, range_x=180, range_y=180, range_z=180),
            transforms.RandGridDistortion(prob=0.5, num_cells=5, distort_limit=0.15, padding_mode="zeros"),
            transforms.RandZoom(prob=0.5, min_zoom=0.75, max_zoom=1.5, mode="trilinear", padding_mode="constant", value=0),
            transforms.RandAffine(prob=0.5, shear_range=0.5, padding_mode="zeros"),
            # transforms.RandAffine(prob=0.5, translate_range=0.5, padding_mode="zeros"),
        ]) if augmentation else transforms.Compose([])

        # Low-Resolution Transforms
        self.lr_transforms = transforms.Compose([
            RandGaussianSmooth(prob=0.5, sigma=(0.0, 1.0)),
            transforms.Zoom(zoom=(1 / scale_factor), mode="trilinear", keep_size=False),
            RandForegroundGaussianNoise(prob=0.5, mean=0.0, std=0.025),
            RandForegroundStreakNoise(prob=0.5, intensity=0.1, line_thickness=(3, 3, 0), num_streaks=(400, 400, 400), origins=(True, True, False))
        ])

        # High-Resolution Transforms
        self.hr_transforms = transforms.Compose([
        ])

        # Revert Low-Resolution Transform
        self.upsample = transforms.Zoom(zoom=scale_factor, mode="trilinear", keep_size=False)

        # Revert Normalization Transform
        self.denormalize = transforms.NormalizeIntensity(subtrahend=-mean/std, divisor=1/std) if (mean and std) else transforms.Identity()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Set Augmentation and Low-Resolution Transforms Seed
        self.augmentation_transforms.set_random_state(seed=(index + self.epoch * self.__len__()))
        self.lr_transforms.set_random_state(seed=(index + self.epoch * self.__len__()))

        # Load Volume
        volume = self.initial_transforms(get_volume_data(os.path.join(self.data_dir, self.files[index])))

        # Employ Data Augmentation
        volume = self.augmentation_transforms(volume)

        # Transform Low-Resolution Volume
        lr_volume = self.lr_transforms(volume)

        # High-Resolution Volume
        hr_volume = self.hr_transforms(volume)

        return self.final_transforms(lr_volume), self.final_transforms(hr_volume)

    def get_low_resolution_shape(self):
        lr_volume, _ = self[0]

        return lr_volume.shape

    def get_high_resolution_shape(self):
        _, hr_volume = self[0]

        return hr_volume.shape

    def increment_epoch(self):
        self.epoch += 1

    def update_epoch(self, epoch):
        self.epoch = epoch

    def upsample(self, volume):
        return self.upsample(volume)

    def denormalize(self, volume):
        return self.denormalize(volume)

    @staticmethod
    def compute_data_statistics(data_dir, batch_size):
        print("=" * 115)

        _min = float('+inf')
        _max = float('-inf')
        mean = np.zeros(1)
        std = np.zeros(1)

        # Load training data
        dataset = SuperResolutionDataset(data_dir=data_dir, scale_factor=1)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Define progress bar
        pbar = tqdm(dataloader, desc="Computing Data Statistics")

        for index, (_, hr_volumes) in enumerate(pbar):
            _min = min(_min, np.min(hr_volumes))
            _max = max(_max, np.max(hr_volumes))
            mean += np.mean(hr_volumes)
            std += np.std(hr_volumes)

            if index == len(pbar) - 1:
                mean = mean.item() / len(dataloader)
                std = std.item() / len(dataloader)
                pbar.set_postfix({"Min": f"{_min:.4f}", "Max": f"{_max:.4f}", "Mean": f"{mean:.4f}", "STD": f"{std:.4f}"})

        # Logging the progress bar
        logging.info(pbar)

        return _min, _max, mean, std


if __name__ == '__main__':
    pass

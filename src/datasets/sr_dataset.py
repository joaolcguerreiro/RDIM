import os
import sys
import monai.transforms as transforms
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.join('..', '..'))
from src.utils.utils import get_volume_data, get_metadata
from src.utils.transforms import RandGaussianSmooth, RandForegroundGaussianNoise, RandForegroundStreakNoise, RandSpatialCropPair


class SuperResolutionDataset(Dataset):
    def __init__(self, data_dir, scale_factor=None, lr_patch_size=(-1, -1, -1), _min=None, _max=None, mean=None, std=None, augmentation=False, downsample=True):
        super().__init__()

        # Data Directory Path
        self.data_dir = data_dir

        # Files
        self.files = os.listdir(data_dir)

        # Track Epoch Count
        self.epoch = 0

        # Common Transforms
        self.initial_transforms = transforms.Compose([
            transforms.EnsureChannelFirst(channel_dim='no_channel'),    # TODO MAKE MY OWN TRANSFORM? GUARANTEE CHANNEL DIM
            transforms.ThresholdIntensity(threshold=(_min-1), above=True, cval=_min) if _min is not None else transforms.Identity(),
            transforms.ThresholdIntensity(threshold=(_max+1), above=False, cval=_min) if (_min is not None and _max is not None) else transforms.Identity(),
        ])

        self.final_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.NormalizeIntensity(subtrahend=mean, divisor=std) if (mean is not None and std is not None) else transforms.Identity(),
        ])

        # Data Augmentation Transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.NormalizeIntensity(subtrahend=_min, divisor=(_max-_min)) if (_min is not None and _max is not None) else transforms.Identity(),
            transforms.RandFlip(prob=0.5),
            transforms.RandRotate(prob=0.5, range_x=180, range_y=180, range_z=180, padding_mode="zeros"),
            transforms.RandGridDistortion(prob=0.5, num_cells=5, distort_limit=0.15, padding_mode="zeros"),
            transforms.RandZoom(prob=0.5, min_zoom=0.75, max_zoom=1.5, mode="trilinear", padding_mode="constant", value=0),
            transforms.RandAffine(prob=0.5, shear_range=0.5, padding_mode="zeros"),
            # TODO transforms.RandAffine(prob=0.5, translate_range=0.5, padding_mode="zeros"),
            transforms.NormalizeIntensity(subtrahend=-_min/(_max-_min), divisor=1/(_max-_min)) if (_min is not None and _max is not None) else transforms.Identity(),
        ]) if augmentation else transforms.Compose([])

        # Patch-based computation transform
        self.patch_transform = RandSpatialCropPair(lr_roi_size=lr_patch_size, scale_factor=scale_factor)

        # Low-Resolution Transforms # TODO REMOVE THIS TRANSFORMS
        self.lr_transforms = transforms.Compose([
            # TODO REMOVE DOWNSAMPLING TRANSFORMS
            RandGaussianSmooth(prob=0.5, sigma=(0.0, 1.0)),
            transforms.Zoom(zoom=(1/scale_factor), mode="trilinear", keep_size=False) if scale_factor else transforms.Identity(),
            RandForegroundGaussianNoise(prob=0.5, mean=0.01, std=0.03),
            RandForegroundStreakNoise(prob=0.5, intensity=1.1, line_thickness=1, num_origins=(40, 90), num_streaks=(60, 120), origin_radius_ratio=1.5, slope_variation=0.7, white_noise=False),
        ]) if downsample else transforms.Compose([])

        # High-Resolution Transforms # TODO REMOVE THIS TRANSFORMS
        self.hr_transforms = transforms.Compose([
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Set Augmentation, Low-Resolution and Patch Transforms Seed
        self.augmentation_transforms.set_random_state(seed=(index + self.epoch * self.__len__()))
        self.lr_transforms.set_random_state(seed=(index + self.epoch * self.__len__()))
        self.patch_transform.set_random_state(seed=(index + self.epoch * self.__len__()))

        # Define volume path
        volume_path = os.path.join(self.data_dir, self.files[index])

        # Load Volume and Metadata
        volume = self.initial_transforms(get_volume_data(volume_path))
        volume_metadata = get_metadata(volume_path)

        # Employ Data Augmentation
        volume = self.augmentation_transforms(volume)

        # Transform Low-Resolution Volume
        lr_volume = self.lr_transforms(volume)

        # High-Resolution Volume
        hr_volume = self.hr_transforms(volume)

        # Employ Patch Crop
        lr_volume, hr_volume = self.patch_transform(lr_volume=lr_volume, hr_volume=hr_volume)

        return self.final_transforms(lr_volume), self.final_transforms(hr_volume), volume_metadata

    def get_low_resolution_shape(self):
        lr_volume, _, _ = self[0]

        return lr_volume.shape

    def get_high_resolution_shape(self):
        _, hr_volume, _ = self[0]

        return hr_volume.shape

    def increment_epoch(self):
        self.epoch += 1

    def update_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def sr_dataset_collate(batch):
        # Unpack data which is a list of tuples
        lr_volumes, hr_volumes, volumes_metadata = zip(*batch)

        # Stack the lr_volumes and hr_volumes
        lr_volumes = torch.stack(lr_volumes, 0)
        hr_volumes = torch.stack(hr_volumes, 0)

        # Combine metadata into one dict
        volumes_metadata = {key: [_dict[key] for _dict in volumes_metadata] for key in next(iter(volumes_metadata or []), {})}

        return lr_volumes, hr_volumes, volumes_metadata


if __name__ == '__main__':
    pass

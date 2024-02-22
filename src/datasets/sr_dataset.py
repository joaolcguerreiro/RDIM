import os
import sys
import math
import monai.transforms as transforms
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.join('..', '..'))
from src.utils.utils import get_volume_data, get_volume_metadata, compute_super_resolution_volume_metadata
from src.utils.transforms import EnsureChannelExists, ThresholdIntensity, NormalizeIntensity, RandSpatialCrop, RandTranslate


class SuperResolutionDataset(Dataset):
    def __init__(self, lr_data_dir=None, hr_data_dir=None, scale_factor=None, lr_patch_size=(-1, -1, -1), _min=None, _max=None, mean=None, std=None, augmentation=False):
        super().__init__()

        # Data Directory Paths
        self.lr_data_dir = lr_data_dir
        self.hr_data_dir = hr_data_dir

        # Files
        self.lr_files = sorted(os.listdir(lr_data_dir)) if lr_data_dir else None
        self.hr_files = sorted(os.listdir(hr_data_dir)) if hr_data_dir else None

        # Resolution ratio between LR and HR
        self.scale_factor = scale_factor

        # Track Epoch Count
        self.epoch = 0

        # Common Transforms
        self.initial_transforms = transforms.Compose([
            EnsureChannelExists(),
            ThresholdIntensity(threshold=_min, above=True, cval=_min, channel_wise=True) if _min is not None else transforms.Identity(),
            ThresholdIntensity(threshold=_max, above=False, cval=_max, channel_wise=True) if _max is not None else transforms.Identity(),
        ])

        self.final_transforms = transforms.Compose([
            NormalizeIntensity(mean=mean, std=std, channel_wise=True) if (mean is not None and std is not None) else transforms.Identity(),
            transforms.ToTensor(),
        ])

        # Data Augmentation Transforms
        self.augmentation_transforms = transforms.Compose([
            NormalizeIntensity(mean=_min, std=(_max-_min), channel_wise=True) if (_min is not None and _max is not None) else transforms.Identity(),
            transforms.RandFlip(prob=0.25),
            transforms.RandRotate(prob=0.25, range_x=math.radians(30), range_y=math.radians(30), range_z=math.radians(30), padding_mode="zeros"),
            transforms.RandGridDistortion(prob=0.25, num_cells=5, distort_limit=0.15, padding_mode="zeros"),
            transforms.RandZoom(prob=0.25, min_zoom=0.85, max_zoom=1.15, mode="trilinear", padding_mode="constant", value=0),
            transforms.RandAffine(prob=0.25, shear_range=0.5, padding_mode="zeros"),
            RandTranslate(prob=0.25, translate_range=0.15, padding_mode="zeros"),
            NormalizeIntensity(mean=-_min/(_max-_min), std=1/(_max-_min), channel_wise=True) if (_min is not None and _max is not None) else transforms.Identity(),
        ]) if augmentation else transforms.Compose([])

        # Patch-based computation transform
        self.patch_transform = RandSpatialCrop(lr_roi_size=lr_patch_size)

    def __len__(self):
        return len(self.lr_files) if self.lr_data_dir else (len(self.hr_files) if self.hr_data_dir else 0)

    def __getitem__(self, index):
        # Set Transforms Seed
        seed = index + self.epoch * self.__len__()

        # Define volumes path
        lr_volume_path = os.path.join(self.lr_data_dir, self.lr_files[index]) if self.lr_data_dir else None
        hr_volume_path = os.path.join(self.hr_data_dir, self.hr_files[index]) if self.hr_data_dir else None

        # Load Volumes and Super-Resolution Metadata
        lr_volume = self.initial_transforms(get_volume_data(lr_volume_path)) if self.lr_data_dir else None
        hr_volume = self.initial_transforms(get_volume_data(hr_volume_path)) if self.hr_data_dir else None

        sr_metadata = get_volume_metadata(hr_volume_path) if self.hr_data_dir else (compute_super_resolution_volume_metadata(lr_metadata=get_volume_metadata(lr_volume_path), scale_factor=self.scale_factor) if self.lr_data_dir else None)

        # Employ Data Augmentation
        lr_volume = self.apply_random_transform(transform=self.augmentation_transforms, volume=lr_volume, seed=seed) if self.lr_data_dir else None
        hr_volume = self.apply_random_transform(transform=self.augmentation_transforms, volume=hr_volume, seed=seed) if self.hr_data_dir else None

        # Employ Patch Crop
        lr_volume = self.apply_random_transform(transform=self.patch_transform, volume=lr_volume, seed=seed, scale_factor=1) if self.lr_data_dir else None
        hr_volume = self.apply_random_transform(transform=self.patch_transform, volume=hr_volume, seed=seed, scale_factor=self.scale_factor) if self.hr_data_dir else None

        # Apply Final Transforms
        lr_volume = self.final_transforms(lr_volume) if self.lr_data_dir else None
        hr_volume = self.final_transforms(hr_volume) if self.hr_data_dir else None

        return lr_volume, hr_volume, sr_metadata

    def get_low_resolution_shape(self, only_spatial_dims=False):
        lr_volume, _, _ = self[0]

        return lr_volume.shape[(1 if only_spatial_dims else None):] if self.lr_data_dir else None

    def get_high_resolution_shape(self, only_spatial_dims=False):
        _, hr_volume, _ = self[0]

        return hr_volume.shape[(1 if only_spatial_dims else None):] if self.hr_data_dir else None

    def get_num_channels(self):
        lr_volume, hr_volume, _ = self[0]

        return lr_volume.shape[0] if self.lr_data_dir else (hr_volume.shape[0] if self.hr_data_dir else None)

    def increment_epoch(self):
        self.epoch += 1

    def update_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def apply_random_transform(transform, volume, seed, **kwargs):
        transform.set_random_state(seed=seed)

        return transform(volume, **kwargs)

    @staticmethod
    def sr_dataset_collate(batch):
        # Unpack data which is a list of tuples
        lr_volumes, hr_volumes, volumes_metadata = zip(*batch)

        # Stack the lr_volumes and hr_volumes (expects either all None or all not None)
        lr_volumes = None if all(lr_volume is None for lr_volume in lr_volumes) else torch.stack(lr_volumes, 0)
        hr_volumes = None if all(hr_volume is None for hr_volume in hr_volumes) else torch.stack(hr_volumes, 0)

        # Combine metadata into one dict (expects either all None or all not None)
        volumes_metadata = None if all(metadata is None for metadata in volumes_metadata) else {key: [_dict[key] for _dict in volumes_metadata] for key in next(iter(volumes_metadata or []), {})}

        return lr_volumes, hr_volumes, volumes_metadata


if __name__ == '__main__':
    pass

import os
import monai.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ..utils.utils import get_volume_data


class SuperResolutionDataset(Dataset):
    def __init__(self, data_dir, mean, std, scale_factor=2, augmentation=False, normalization=True):
        super(SuperResolutionDataset, self).__init__()

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
            transforms.ThresholdIntensity(threshold=0),
            transforms.SpatialPad(spatial_size=(400, 400, 400)),
            transforms.CenterSpatialCrop(roi_size=(400, 400, 400)),
        ])

        self.final_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.NormalizeIntensity(subtrahend=mean, divisor=std) if normalization else (lambda x: x),
        ])

        # Data Augmentation Transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.RandFlip(prob=0.5),
            transforms.RandRotate(prob=0.5, range_x=180, range_y=180, range_z=180),
            transforms.RandGridDistortion(prob=0.5, num_cells=5, distort_limit=0.15, padding_mode="zeros"),
            transforms.RandAffine(prob=0.5, shear_range=0.5, padding_mode="zeros"),
            transforms.RandAffine(prob=0.5, translate_range=(0.5 * 400), padding_mode="zeros"),
            transforms.RandZoom(prob=0.5, min_zoom=0.75, max_zoom=1.5, mode="trilinear", padding_mode="constant", value=0),
        ]) if augmentation else transforms.Compose([])

        # Low-Resolution Transforms
        self.lr_transforms = transforms.Compose([
            transforms.Zoom(zoom=(1 / scale_factor), mode="trilinear", keep_size=False),
        ])

        # High-Resolution Transforms
        self.hr_transforms = transforms.Compose([
        ])

        # Revert Low-Resolution Transform
        self.upsample = transforms.Zoom(zoom=scale_factor, mode="trilinear", keep_size=False)

        # Revert Normalization Transform
        self.denormalize = transforms.NormalizeIntensity(subtrahend=-mean/std, divisor=1/std)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load Volume
        volume = self.initial_transforms(get_volume_data(os.path.join(self.data_dir, self.files[index])))

        # Set Augmentation Transforms Seed
        self.augmentation_transforms.set_random_state(seed=(index + self.epoch * self.__len__()))

        # Employ Data Augmentation
        volume = self.augmentation_transforms(volume)

        # Transform Low-Resolution Volume
        lr_volume = self.lr_transforms(volume)

        # High-Resolution Volume
        hr_volume = self.hr_transforms(volume)

        return self.final_transforms(lr_volume), self.final_transforms(hr_volume)

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

        mean = np.zeros(1)
        std = np.zeros(1)

        # Load training data
        dataset = SuperResolutionDataset(data_dir=data_dir, mean=-1, std=-1, scale_factor=1, normalization=False, augmentation=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define progress bar
        pbar = tqdm(dataloader, desc="Computing Mean and STD")

        for index, (_, hr_volumes) in enumerate(pbar):
            mean += np.mean(hr_volumes)
            std += np.std(hr_volumes)

            if index == len(pbar) - 1:
                mean = mean.item() / len(dataloader)
                std = std.item() / len(dataloader)
                pbar.set_postfix({"Mean": f"{mean:.4f}", "STD": f"{std:.4f}"})

        return mean, std


if __name__ == '__main__':
    pass

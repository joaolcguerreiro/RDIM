import os
import monai.transforms as transforms
from torch.utils.data import Dataset
from ..utils.utils import get_volume_data


class SuperResolutionBraTSDataset(Dataset):
    def __int__(self, data_dir, scale_factor=2, augmentation=False):
        super(SuperResolutionBraTSDataset, self).__int__()

        # Data Directory Path
        self.data_dir = data_dir

        # Files
        self.files = os.listdir(data_dir)

        # Scale Factor
        self.scale_factor = scale_factor

        # Data Augmentation Transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.AddChanneld(),
            transforms.RandFlip(prob=0.5),
            transforms.RandRotate(prob=0.5, range_x=180, range_y=180, range_z=180),
            transforms.RandZoom(prob=0.5, min_zoom=0.15, max_zoom=0.15),
            transforms.RandAffine(prob=0.5, shear_range=1.0),
            transforms.RandGridDistortion(prob=0.5, num_cells=5, distort_limit=0.1),
            transforms.ToTensor()
        ]) if augmentation else transforms.Compose([])

        # Low-Resolution Transforms
        self.lr_transforms = transforms.Compose([
            transforms.Zoom(zoom=(1 / scale_factor), mode="bicubic", keep_size=False),
            transforms.ToTensor()
        ])

        # High-Resolution Transforms
        self.hr_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load Volume
        volume = get_volume_data("flair", os.path.join(self.data_dir, self.files[index]))

        # Employ Data Augmentation
        volume = self.augmentation_transforms(volume)

        # Transform Low-Resolution Volume
        lr_volume = self.lr_transforms(volume)

        # High-Resolution Volume
        hr_volume = self.hr_transforms(volume)

        return lr_volume, hr_volume


if __name__ == '__main__':
    pass

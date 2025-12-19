import os
import sys
import torch
import torch.nn as nn
import monai.transforms as transforms
from monai import losses
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.utils import denormalize, min_max_scaling
from src.utils.transforms import EnsureExactChannels


class LPIPSLoss(nn.Module):
    def __init__(self, num_spatial_dimensions, _min, _max, mean=None, std=None):
        super().__init__()

        # Define Criterion
        self.criterion = losses.PerceptualLoss(spatial_dims=num_spatial_dimensions, network_type='alex')

        # AlexNet transforms
        self.alex_net_transforms = transforms.Compose([
            EnsureExactChannels(num_channels=3, num_spatial_dimensions=num_spatial_dimensions),
            transforms.ScaleIntensityRange(a_min=_min, a_max=_max, b_min=-1.0, b_max=1.0),
            transforms.ToTensor(),
        ])

        # Define statistics
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([self.alex_net_transforms(denormalize(prediction, mean=self.mean, std=self.std)) for prediction in predictions], dim=0)
        ground_truths = torch.stack([self.alex_net_transforms(denormalize(ground_truth, mean=self.mean, std=self.std)) for ground_truth in ground_truths], dim=0)

        return self.criterion(predictions, ground_truths)


class SSIMLoss(nn.Module):
    def __init__(self, num_spatial_dimensions, _min, _max, mean=None, std=None):
        super().__init__()

        # Define Criterion
        self.criterion = losses.SSIMLoss(spatial_dims=num_spatial_dimensions, data_range=1.0)

        # Define statistics
        self.register_buffer('_min', _min)
        self.register_buffer('_max', _max)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions], dim=0)
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths], dim=0)

        return self.criterion(predictions, ground_truths)

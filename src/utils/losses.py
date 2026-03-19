import os
import sys
import warnings
import torch
import torch.nn as nn
import monai.transforms as transforms
from monai import losses
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.utils import denormalize, min_max_scaling
from src.utils.transforms import EnsureExactChannels


class CharbonnierLoss(nn.Module):
    def __init__(self, _min=None, _max=None, mean=None, std=None, eps=1e-2):
        super().__init__()

        # Define Criterion
        self.criterion = partial(self.compute_charbonnier_loss, eps=eps)

        # Define statistics
        self._min = _min
        self._max = _max
        self.mean = mean
        self.std = std

    @staticmethod
    def compute_charbonnier_loss(predictions, ground_truths, eps):
        return torch.mean(torch.sqrt(torch.pow(ground_truths - predictions, 2) + eps ** 2))

    def forward(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions], dim=0)
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths], dim=0)

        return self.criterion(predictions, ground_truths)


class LPIPSLoss(nn.Module):
    def __init__(self, num_spatial_dimensions, _min, _max, mean=None, std=None):
        super().__init__()

        # Suppress the specific warning related to pretrained model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
            warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
            warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

            # Define Criterion
            self.criterion = losses.PerceptualLoss(spatial_dims=num_spatial_dimensions, network_type='alex')

        # AlexNet transforms
        self.alex_net_transforms = transforms.Compose([
            EnsureExactChannels(num_channels=3, num_spatial_dimensions=num_spatial_dimensions),
            transforms.ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0),
            transforms.ToTensor(),
        ])

        # Define statistics
        self._min = _min
        self._max = _max
        self.mean = mean
        self.std = std

    def forward(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([self.alex_net_transforms(min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max)) for prediction in predictions], dim=0)
        ground_truths = torch.stack([self.alex_net_transforms(min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max)) for ground_truth in ground_truths], dim=0)

        return self.criterion(predictions, ground_truths)


class SSIMLoss(nn.Module):
    def __init__(self, num_spatial_dimensions, _min, _max, mean=None, std=None):
        super().__init__()

        # Define Criterion
        self.criterion = losses.SSIMLoss(spatial_dims=num_spatial_dimensions, data_range=1.0)

        # Define statistics
        self._min = _min
        self._max = _max
        self.mean = mean
        self.std = std

    def forward(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions], dim=0)
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths], dim=0)

        return self.criterion(predictions, ground_truths)

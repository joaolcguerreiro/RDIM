import os
import sys
import torch
import numpy as np
import torch.nn as nn
import monai.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from monai import losses
from monai.inferers.inferer import PatchInferer
from monai.inferers import SlidingWindowSplitter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.utils import parse_patch_size, denormalize, min_max_scaling
from src.utils.transforms import NormalizeIntensity, EnsureExactChannels


class VGGLoss(nn.Module):
    """
    Computes perceptual loss. For 3D settings, a 2.5D VGG Loss is computed by averaging the 2D perceptual loss values on slices from all three axes.
    Expected volume dimensions order is arbitrary.
    """

    def __init__(self, num_spatial_dimensions, _min, _max, mean=None, std=None, volume_size=None, patch_size=None, min_spatial_size=None):
        super().__init__()
        assert not (volume_size is None and (np.any(np.array(patch_size) < 0) or patch_size is None)), f'Volume size is required when any patch size dimension is unknown (at indexes: {np.where(np.array(patch_size) < 0)[0]})'

        if patch_size is None:
            patch_size = (-1,) * num_spatial_dimensions

        if min_spatial_size is None:
            min_spatial_size = (16,) * num_spatial_dimensions

        # Define Criterion
        self.criterion = nn.L1Loss()

        # VGG Network as feature extractor
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35]

        # Set the model to eval mode
        self.vgg.eval()

        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

        # VGG transforms
        self.vgg_transforms = transforms.Compose([
            NormalizeIntensity(mean=-mean/std, std=1/std, channel_wise=True) if (mean is not None and std is not None) else transforms.Identity(),
            NormalizeIntensity(mean=_min, std=(_max-_min), channel_wise=True),
            transforms.SpatialPad(spatial_size=min_spatial_size),
            EnsureExactChannels(num_channels=3, num_spatial_dimensions=num_spatial_dimensions),
            NormalizeIntensity(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel_wise=True),
            transforms.ToTensor(),
        ])

        # Set patch size
        self.patch_size = np.maximum(parse_patch_size(patch_size, volume_size), min_spatial_size)

        # Define processing planes
        self.planes = [
            {'orthogonal_axis_dim': -3, 'coplanar_axes_dims': [-2, -1]},
            {'orthogonal_axis_dim': -2, 'coplanar_axes_dims': [-3, -1]},
            {'orthogonal_axis_dim': -1, 'coplanar_axes_dims': [-3, -2]},
        ] if num_spatial_dimensions == 3 else [
            {'coplanar_axes_dims': [-2, -1]},
        ]

    def forward(self, predictions, ground_truths):
        # Preprocess data as expected by VGG
        transformed_predictions = torch.stack([self.vgg_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truths = torch.stack([self.vgg_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Define total VGG loss
        vgg_total_loss = 0.0

        # Handle plane-wise processing (2.5D)
        for plane in self.planes:
            # Define network inferer
            inferer = PatchInferer(splitter=SlidingWindowSplitter(patch_size=list(self.patch_size[dim] for dim in plane['coplanar_axes_dims']), overlap=0.125, device=next(self.parameters()).device), match_spatial_shape=True, device=next(self.parameters()).device)

            # Convert 3D data to 2D
            transformed_predictions_2d = transformed_predictions.permute(-5, plane['orthogonal_axis_dim'], -4, *plane['coplanar_axes_dims']).contiguous().view(-1, 3, *(transformed_predictions.shape[dim] for dim in plane['coplanar_axes_dims'])) if transformed_predictions.dim() == 5 else transformed_predictions
            transformed_ground_truths_2d = transformed_ground_truths.permute(-5, plane['orthogonal_axis_dim'], -4, *plane['coplanar_axes_dims']).contiguous().view(-1, 3, *(transformed_ground_truths.shape[dim] for dim in plane['coplanar_axes_dims'])) if transformed_ground_truths.dim() == 5 else transformed_ground_truths

            # Compute VGG features
            vgg_features_predictions_2d = inferer(inputs=transformed_predictions_2d, network=self.vgg)
            vgg_features_ground_truths_2d = inferer(inputs=transformed_ground_truths_2d, network=self.vgg).detach()

            # Accumulate and compute plane VGG loss
            vgg_total_loss += self.criterion(vgg_features_predictions_2d, vgg_features_ground_truths_2d)

        return vgg_total_loss / len(self.planes)


class SSIMLoss(nn.Module):
    def __init__(self, num_spatial_dims, _min, _max, mean=None, std=None):
        super().__init__()

        # Define Criterion
        self.criterion = losses.SSIMLoss(spatial_dims=num_spatial_dims, data_range=1.0)

        # Define statistics
        self._min = _min
        self._max = _max
        self.mean = mean
        self.std = std

    def to(self, *args, **kwargs):
        # Required to move non-parameter attributes to device, change their dtype, and more
        self._min = self._min.to(*args, **kwargs)
        self._max = self._max.to(*args, **kwargs)
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def forward(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions], dim=0)
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths], dim=0)

        return self.criterion(predictions, ground_truths)

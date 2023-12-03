import os
import sys
import torch
import torch.nn as nn
import monai.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from monai.inferers.inferer import SlidingWindowInferer
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import NormalizeIntensity, EnsureExactChannels
from src.models.esrgan import Discriminator


class VGGLoss(nn.Module):
    def __init__(self, _min, _max, mean=None, std=None, patch_size=(-1, -1, -1)):
        super().__init__()

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
            transforms.SpatialPad(spatial_size=(32, 32, 32)),
            EnsureExactChannels(num_channels=3),
            transforms.ToTensor(),
            NormalizeIntensity(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel_wise=True),
        ])

        # Define network inferer
        self.inferer = SlidingWindowInferer(roi_size=patch_size[-3:-1], overlap=0.25, mode='gaussian')

    def forward(self, predictions, ground_truths):
        # Preprocess data as expected by VGG
        transformed_predictions = torch.stack([self.vgg_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truths = torch.stack([self.vgg_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Convert 3D data to 2D
        transformed_predictions_2d = transformed_predictions.view(-1, 3, *(transformed_predictions.shape[-3:-1]))
        transformed_ground_truths_2d = transformed_ground_truths.view(-1, 3, *(transformed_ground_truths.shape[-3:-1]))

        # Compute VGG features
        vgg_features_predictions = self.inferer(inputs=transformed_predictions_2d, network=self.vgg)
        vgg_features_ground_truths = self.inferer(inputs=transformed_ground_truths_2d, network=self.vgg).detach()

        return self.criterion(vgg_features_predictions, vgg_features_ground_truths)


class GANDLoss(nn.Module):
    """
    GANDLoss (Generative Adversarial Network Discriminator Loss), computes perceptual loss based on the features from a GAN discriminator.
    """

    def __init__(self, checkpoint, mean, std, patch_size=(-1, -1, -1)):
        super().__init__()

        # Define Criterion
        self.criterion = nn.L1Loss()

        # GAN Discriminator Network as feature extractor
        self.discriminator = Discriminator(
            in_channels=checkpoint['discriminator_in_channels'],
            num_features=checkpoint['discriminator_num_features'],
            num_conv_blocks=checkpoint['discriminator_num_conv_blocks'],
            conv_blocks_output_size=checkpoint['discriminator_conv_blocks_output_size']
        )

        # Initialize Feature Extractor weights
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Replace forward method to extract features
        self.discriminator.forward = self.discriminator.extract_features

        # Set the model to eval mode
        self.discriminator.eval()

        for parameter in self.discriminator.parameters():
            parameter.requires_grad = False

        # Discriminator transforms
        self.discriminator_transforms = transforms.Compose([
            NormalizeIntensity(mean=-mean/std, std=1/std, channel_wise=True),
            EnsureExactChannels(num_channels=checkpoint['discriminator_num_features']),
            transforms.ToTensor(),
            NormalizeIntensity(mean=checkpoint['training_data_mean'], std=checkpoint['training_data_std'], channel_wise=True),
        ])

        # Define network inferer
        self.inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.25, mode='gaussian')

    def forward(self, predictions, ground_truths):
        # Preprocess data as expected by Discriminator
        transformed_predictions = torch.stack([self.discriminator_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truths = torch.stack([self.discriminator_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Compute Discriminator features
        discriminator_features_predictions = self.inferer(inputs=transformed_predictions, network=self.discriminator)
        discriminator_features_ground_truths = self.inferer(inputs=transformed_ground_truths, network=self.discriminator).detach()

        return self.criterion(discriminator_features_predictions, discriminator_features_ground_truths)


class RangeLoss(nn.Module):
    """
    Compute loss that constraints generated data to be within a specific range by penalizing outputs outside it.
    """

    def __init__(self, _min, _max, mean=None, std=None):
        super().__init__()

        # Define Criterion
        self.criterion = nn.L1Loss()

        # Define data range
        self._min = torch.tensor(_min if (mean is None or std is None) else ((_min - mean) / std)).view(1, -1, *[1] * 3)
        self._max = torch.tensor(_max if (mean is None or std is None) else ((_max - mean) / std)).view(1, -1, *[1] * 3)

    def forward(self, predictions):
        below = torch.relu(self._min - predictions)
        above = torch.relu(predictions - self._max)

        return torch.mean(below + above)

import os
import sys
import torch
import statistics
import numpy as np
import torch.nn as nn
import monai.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from monai.inferers.inferer import SlidingWindowInferer
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import NormalizeIntensity, EnsureExactChannels
from src.models.esrgan import Discriminator


class VGGLoss(nn.Module):
    def __init__(self, _min, _max, mean, std, patch_size=(-1, -1, -1)):
        super().__init__()

        # Define Criterion
        self.criterion = nn.L1Loss()

        # VGG Network as feature extractor
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35]

        # Set the model to eval mode
        self.vgg.eval()

        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

        # Define model statistics
        mean_model, std_model = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # VGG transforms
        self.vgg_transforms = transforms.Compose([
            NormalizeIntensity(mean=-mean/std, std=1/std, channel_wise=True),
            NormalizeIntensity(mean=_min, std=(_max-_min), channel_wise=True),
            transforms.SpatialPad(spatial_size=(32, 32, 32)),
            EnsureExactChannels(num_channels=3),
            transforms.ToTensor(),
            NormalizeIntensity(mean=mean_model, std=std_model, channel_wise=True),
        ])

        # Compute normalized minimum
        min_normalized = statistics.fmean((np.min(np.zeros_like(_min)) - np.mean(mean_model)) / np.mean(std_model))

        # Define network inferer
        self.inferer = SlidingWindowInferer(roi_size=patch_size[-3:-1], overlap=0.25, mode='gaussian', cval=min_normalized)

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

    def __init__(self, checkpoint, _min, mean, std, patch_size=(-1, -1, -1)):
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

        # Fetch model statistics
        mean_model, std_model = checkpoint['training_data_mean'], checkpoint['training_data_std']

        # Discriminator transforms
        self.discriminator_transforms = transforms.Compose([
            NormalizeIntensity(mean=-mean/std, std=1/std, channel_wise=True),
            EnsureExactChannels(num_channels=checkpoint['discriminator_num_features']),
            transforms.ToTensor(),
            NormalizeIntensity(mean=mean_model, std=std_model, channel_wise=True),
        ])

        # Compute normalized minimum
        min_normalized = statistics.fmean((np.min(_min) - np.mean(mean_model)) / np.mean(std_model))

        # Define network inferer
        self.inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.25, mode='gaussian', cval=min_normalized)

    def forward(self, predictions, ground_truths):
        # Preprocess data as expected by Discriminator
        transformed_predictions = torch.stack([self.discriminator_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truths = torch.stack([self.discriminator_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Compute Discriminator features
        discriminator_features_predictions = self.inferer(inputs=transformed_predictions, network=self.discriminator)
        discriminator_features_ground_truths = self.inferer(inputs=transformed_ground_truths, network=self.discriminator).detach()

        return self.criterion(discriminator_features_predictions, discriminator_features_ground_truths)

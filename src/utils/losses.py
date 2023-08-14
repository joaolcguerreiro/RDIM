import os
import sys
import torch
import torch.nn as nn
import monai.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from monai.inferers.inferer import SlidingWindowInferer
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import NormalizeIntensity, EnsureExactChannels


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

        # VGG transforms
        self.vgg_transforms = transforms.Compose([
            NormalizeIntensity(mean=-mean/std, std=1/std, channel_wise=True),
            NormalizeIntensity(mean=_min, std=(_max-_min), channel_wise=True),
            transforms.SpatialPad(spatial_size=(32, 32, 32)),
            EnsureExactChannels(num_channels=3),
            transforms.ToTensor(),
            NormalizeIntensity(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel_wise=True),
        ])

        # Define network inferer
        self.inferer = SlidingWindowInferer(roi_size=patch_size[-3:-1], overlap=0.25, mode='gaussian', cval=_min)

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

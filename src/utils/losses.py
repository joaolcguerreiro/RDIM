import os
import sys
import torch
import torch.nn as nn
import monai.transforms as transforms
from torchvision.models import vgg19
sys.path.append(os.path.join('..', '..'))
from src.utils.transforms import NormalizeIntensity


class VGGLoss(nn.Module):
    def __init__(self, _min, _max, mean, std):
        super().__init__()

        # Define Criterion
        self.criterion = nn.L1Loss()

        # VGG Network as feature extractor
        self.vgg = vgg19(pretrained=True).features[:35]

        # Set the model to eval mode
        self.vgg.eval()

        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

        # VGG transforms
        self.vgg_transforms = transforms.Compose([
            transforms.Resize(spatial_size=(256, 256, 256), mode="trilinear"),
            transforms.CenterSpatialCrop(roi_size=(224, 224, 224)),
            NormalizeIntensity(mean=-mean / std, std=1 / std),
            NormalizeIntensity(mean=_min, std=(_max - _min)),
            transforms.RepeatChannel(repeats=3),
            transforms.ToTensor(),
            NormalizeIntensity(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel_wise=True),
        ])

    def forward(self, predictions, ground_truths):
        # Preprocess data as expected by VGG
        transformed_prediction = torch.stack([self.vgg_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truth = torch.stack([self.vgg_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Convert 3D data to 2D
        transformed_prediction_2d = transformed_prediction.view(-1, 3, 224, 224)
        transformed_ground_truth_2d = transformed_ground_truth.view(-1, 3, 224, 224)

        return self.criterion(self.vgg(transformed_prediction_2d), self.vgg(transformed_ground_truth_2d).detach())

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import monai.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from monai.inferers.inferer import SlidingWindowInferer, PatchInferer
from monai.inferers import SlidingWindowSplitter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.transforms import NormalizeIntensity, EnsureExactChannels
from src.models.esrgan import Discriminator


class VGGLoss(nn.Module):
    """
    2.5D VGG Loss, computes average 2D perceptual loss on slices from all three axes.
    Expected volume dimensions order is arbitrary.
    """

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
            NormalizeIntensity(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel_wise=True),
            transforms.ToTensor(),
        ])

        # Set patch size
        self.patch_size = np.maximum(patch_size, (32, 32, 32))

        # Define processing planes
        self.planes = [
            {'orthogonal_axis_dim': -3, 'coplanar_axes_dims': [-2, -1]},
            {'orthogonal_axis_dim': -2, 'coplanar_axes_dims': [-3, -1]},
            {'orthogonal_axis_dim': -1, 'coplanar_axes_dims': [-3, -2]},
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
            inferer = SlidingWindowInferer(roi_size=list(self.patch_size[dim] for dim in plane['coplanar_axes_dims']), overlap=0.25, mode='gaussian')

            # Convert 3D data to 2D
            transformed_predictions_2d = transformed_predictions.permute(-5, plane['orthogonal_axis_dim'], -4, *plane['coplanar_axes_dims']).contiguous().view(-1, 3, *(transformed_predictions.shape[dim] for dim in plane['coplanar_axes_dims']))
            transformed_ground_truths_2d = transformed_ground_truths.permute(-5, plane['orthogonal_axis_dim'], -4, *plane['coplanar_axes_dims']).contiguous().view(-1, 3, *(transformed_ground_truths.shape[dim] for dim in plane['coplanar_axes_dims']))

            # Compute VGG features
            vgg_features_predictions_2d = inferer(inputs=transformed_predictions_2d, network=self.vgg)
            vgg_features_ground_truths_2d = inferer(inputs=transformed_ground_truths_2d, network=self.vgg).detach()

            # Accumulate and compute plane VGG loss
            vgg_total_loss += self.criterion(vgg_features_predictions_2d, vgg_features_ground_truths_2d)

        return vgg_total_loss / len(self.planes)


class Med3DLoss(nn.Module):
    def __init__(self, original_size, patch_size=(-1, -1, -1), mean=None, std=None):
        super().__init__()

        # Define Criterion
        self.criterion = nn.L1Loss()

        # Avoid HTTP Error 403
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        # Med3D (Pre-trained 3D-ResNet) as feature extractor
        self.med3d = torch.hub.load("warvito/MedicalNet-models", model="medicalnet_resnet50_23datasets", verbose=False)

        # Set the model to eval mode
        self.med3d.eval()

        for parameter in self.med3d.parameters():
            parameter.requires_grad = False

        # Med3D transforms
        self.med3d_transforms = transforms.Compose([
            NormalizeIntensity(mean=mean, std=std, channel_wise=True) if (mean is not None and std is not None) else transforms.Identity(),
            EnsureExactChannels(num_channels=1),
            transforms.ToTensor(),
        ])

        # Define network inferer with minimum patch size of 8x8x8
        self.inferer = PatchInferer(splitter=SlidingWindowSplitter(patch_size=np.maximum(list(map(lambda patch_dim, original_dim: patch_dim if patch_dim > 0 else original_dim, patch_size, original_size)), [8, 8, 8]).tolist(), overlap=0.25), match_spatial_shape=True)

    def to(self, *args, **kwargs):
        # Required to move non-parameter attributes to device
        self.inferer.merger_kwargs = {'device': kwargs.get('device') or next(iter(args), None)}

        return super().to(*args, **kwargs)

    def forward(self, predictions, ground_truths):
        # Preprocess data as expected by Med3D
        transformed_predictions = torch.stack([self.med3d_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truths = torch.stack([self.med3d_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Compute Med3D features
        med3d_features_predictions = self.inferer(inputs=transformed_predictions, network=self.med3d)
        med3d_features_ground_truths = self.inferer(inputs=transformed_ground_truths, network=self.med3d).detach()

        return self.criterion(med3d_features_predictions, med3d_features_ground_truths)


class GANLoss(nn.Module):
    """
    GANLoss (Generative Adversarial Network Loss), computes adversarial loss for GAN framework.
    """

    def __init__(self, _type='STANDARD', with_logits=False):
        super().__init__()
        assert _type == 'STANDARD' if not with_logits else True, f'If discriminator outputs probabilities rather than logits use \'STANDARD\' gan type instead of \'{_type}\' gan type.'

        # Adversarial Loss Type
        self._type = _type

        # Define Criterion for Standard Adversarial Loss types
        self.criterion = nn.BCELoss() if not with_logits else nn.BCEWithLogitsLoss()

    def generator_loss(self, real_labels_pred=None, fake_labels_pred=None, real_labels_true=None, fake_labels_true=None):
        if self._type == 'STANDARD':
            loss_fake = self.criterion(fake_labels_pred, torch.ones_like(real_labels_true))

            return loss_fake
        elif self._type == 'RELATIVISTIC_AVERAGE_STANDARD':
            loss_fake = self.criterion(fake_labels_pred - real_labels_pred.mean(), torch.ones_like(real_labels_true))
            loss_real = self.criterion(real_labels_pred - fake_labels_pred.mean(), torch.zeros_like(fake_labels_true))

            return (loss_fake + loss_real) / 2
        elif self._type == 'RELATIVISTIC_AVERAGE_LEAST_SQUARES':
            loss_fake = torch.mean(torch.pow(fake_labels_pred - real_labels_pred.mean() - real_labels_true, 2))
            loss_real = torch.mean(torch.pow(real_labels_pred - fake_labels_pred.mean() + real_labels_true, 2))

            return (loss_fake + loss_real) / 2
        elif self._type == 'RELATIVISTIC_AVERAGE_HINGE':
            loss_fake = torch.mean(torch.max(fake_labels_true, real_labels_true - (fake_labels_pred - real_labels_pred.mean())))
            loss_real = torch.mean(torch.max(fake_labels_true, real_labels_true + (real_labels_pred - fake_labels_pred.mean())))

            return (loss_fake + loss_real) / 2
        else:
            raise NotImplementedError(f'Invalid GAN loss type: {self._type}.')

    def discriminator_loss(self, real_labels_pred=None, fake_labels_pred=None, real_labels_true=None, fake_labels_true=None):
        if self._type == 'STANDARD':
            loss_real = self.criterion(real_labels_pred, torch.ones_like(real_labels_true))
            loss_fake = self.criterion(fake_labels_pred, torch.zeros_like(fake_labels_true))

            return (loss_real + loss_fake) / 2
        elif self._type == 'RELATIVISTIC_AVERAGE_STANDARD':
            loss_real = self.criterion(real_labels_pred - fake_labels_pred.mean(), torch.ones_like(real_labels_true))
            loss_fake = self.criterion(fake_labels_pred - real_labels_pred.mean(), torch.zeros_like(fake_labels_true))

            return (loss_real + loss_fake) / 2
        elif self._type == 'RELATIVISTIC_AVERAGE_LEAST_SQUARES':
            loss_real = torch.mean(torch.pow(real_labels_pred - fake_labels_pred.mean() - real_labels_true, 2))
            loss_fake = torch.mean(torch.pow(fake_labels_pred - real_labels_pred.mean() + real_labels_true, 2))

            return (loss_real + loss_fake) / 2
        elif self._type == 'RELATIVISTIC_AVERAGE_HINGE':
            loss_real = torch.mean(torch.max(fake_labels_true, real_labels_true - (real_labels_pred - fake_labels_pred.mean())))
            loss_fake = torch.mean(torch.max(fake_labels_true, real_labels_true + (fake_labels_pred - real_labels_pred.mean())))

            return (loss_real + loss_fake) / 2
        else:
            raise NotImplementedError(f'Invalid GAN loss type: {self._type}.')

    def forward(self, real_labels_pred=None, fake_labels_pred=None, real_labels_true=None, fake_labels_true=None, mode='GENERATOR'):
        if mode == 'GENERATOR':
            return self.generator_loss(real_labels_pred=real_labels_pred, fake_labels_pred=fake_labels_pred, real_labels_true=real_labels_true, fake_labels_true=fake_labels_true)
        elif mode == 'DISCRIMINATOR':
            return self.discriminator_loss(real_labels_pred=real_labels_pred, fake_labels_pred=fake_labels_pred, real_labels_true=real_labels_true, fake_labels_true=fake_labels_true)
        else:
            raise NotImplementedError(f'Invalid GAN loss mode: {mode}.')


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
            NormalizeIntensity(mean=checkpoint['training_data_mean'], std=checkpoint['training_data_std'], channel_wise=True),
            transforms.ToTensor(),
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
        self._min = torch.tensor(_min if (mean is None or std is None) else ((_min - mean) / std), requires_grad=False).view(1, -1, *([1] * 3))
        self._max = torch.tensor(_max if (mean is None or std is None) else ((_max - mean) / std), requires_grad=False).view(1, -1, *([1] * 3))

    def to(self, *args, **kwargs):
        # Required to move non-parameter attributes to device, change their dtype, and more
        self._min = self._min.to(*args, **kwargs)
        self._max = self._max.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def forward(self, predictions):
        below = torch.relu(self._min - predictions)
        above = torch.relu(predictions - self._max)

        return torch.mean(below + above)

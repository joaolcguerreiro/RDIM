import os
import sys
import torch
import numpy as np
import lpips
import warnings
import monai.transforms as transforms
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.regression import RegressionMetric
from monai.utils import MetricReduction
from tabulate import tabulate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.logger import print_divisor
from src.utils.utils import denormalize, min_max_scaling
from src.utils.transforms import EnsureExactChannels


class Metrics:
    def __init__(self):
        super().__init__()

        self.metrics = []

    def compute_scores(self, predictions, ground_truths):
        # Preprocessing
        predictions, ground_truths = self.preprocess(predictions=predictions, ground_truths=ground_truths)

        # Compute Y-channel versions
        y_predictions, y_ground_truths = self.rgb_to_y_channel(volumes=predictions), self.rgb_to_y_channel(volumes=ground_truths)

        for metric in self.metrics:
            # Compute metric scores
            if metric['name'].endswith('_Y'):
                scores = metric['func'](y_pred=y_predictions, y=y_ground_truths).squeeze(dim=-1).tolist()
            else:
                scores = metric['func'](y_pred=predictions, y=ground_truths).squeeze(dim=-1).tolist()

            # Store metric score
            metric['scores'] += scores

        return {metric['name']: np.mean(metric['scores'][-len(scores):]) for metric in self.metrics}

    def get_scores(self):
        return {metric['name']: metric['scores'] for metric in self.metrics}

    def summary(self):
        print_divisor()

        print(tabulate([
            [metric['name'], np.mean(metric['scores'])] for metric in self.metrics
        ], headers=['Metric', 'Score']))

        print_divisor()

    def preprocess(self, predictions, ground_truths):
        """
        Method to be optionally overridden by subclasses for preprocessing.
        Default implementation returns the inputs unchanged.
        """

        return predictions, ground_truths

    def rgb_to_y_channel(self, volumes):
        """
        Convert RGB volumes in [0, 1] to Y channel in YCbCr, matching the common MATLAB evaluation convention.
        """

        if volumes.shape[1] == 1:
            return volumes

        if volumes.shape[1] < 3:
            raise ValueError(f"Expected at least 3 channels for RGB conversion to Y, got {volumes.shape[1]}.")

        r = volumes[:, 0:1, ...]
        g = volumes[:, 1:2, ...]
        b = volumes[:, 2:3, ...]

        # BT.601-style | Y = (65.481 R + 128.553 G + 24.966 B) / 255 + 16/255
        y_volumes = (65.481 * r + 128.553 * g + 24.966 * b) / 255.0 + 16.0 / 255.0

        return y_volumes

    def to(self, *args, **kwargs):
        # Required to move attributes to device, change their dtype, and more
        for metric in self.metrics:
            if hasattr(metric['func'], 'to'):
                metric['func'].to(*args, **kwargs)

        return self


class SuperResolutionMetrics(Metrics):
    def __init__(self, num_spatial_dimensions, _min, _max, mean=None, std=None):
        super().__init__()

        # Define statistics
        self._min = _min
        self._max = _max
        self.mean = mean
        self.std = std

        # Define metrics
        self.metrics = [
            {'name': 'PSNR', 'scores': [], 'func': PSNRMetric(max_val=1.0)},
            {'name': 'SSIM', 'scores': [], 'func': SSIMMetric(spatial_dims=num_spatial_dimensions, data_range=1.0)},
            {'name': 'LPIPS', 'scores': [], 'func': LPIPSMetric(num_spatial_dimensions=num_spatial_dimensions, min_val=0.0, max_val=1.0)},
            {'name': 'PSNR_Y', 'scores': [], 'func': PSNRMetric(max_val=1.0)},
            {'name': 'SSIM_Y', 'scores': [], 'func': SSIMMetric(spatial_dims=num_spatial_dimensions, data_range=1.0)},
        ]

    def preprocess(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions])
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths])

        return predictions, ground_truths


class LPIPSMetric(RegressionMetric):
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS) between two tensors using the LPIPS model.

    LPIPS is a perceptual metric that is used to compare the similarity between two images. It uses a deep network to
    extract features and then compares these features in a perceptually meaningful way.

    More info: https://github.com/richzhang/PerceptualSimilarity

    Input `predictions` is compared with ground truth `ground_truths`.

    Args:
        num_spatial_dimensions: Number of spatial dimensions in the tensors.
        min_val: Minimum pixel intensity value.
        max_val: Maximum pixel intensity value.
        reduction: Define the mode to reduce metrics, available reduction modes: {``"none"``, ``"mean"``, ``"sum"``,
            ``"mean_batch"``, ``"sum_batch"``, ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. If "none", will not do reduction.
        get_not_nans: Whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
        model: The backbone network for LPIPS. Default is 'alex' (other options are 'vgg', 'squeeze').
        version: The version of LPIPS. Default is '0.1'.
        pretrained: Use pretrained weights for LPIPS model. Default is True.
    """

    def __init__(self, num_spatial_dimensions: int, min_val: float, max_val: float, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False, model: str = 'alex', version: str = '0.1', pretrained: bool = True) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        # Suppress the specific warning related to pretrained model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
            warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
            warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)

            # Define Criterion
            self.criterion = lpips.LPIPS(net=model, version=version, pretrained=pretrained, verbose=False)

        # Network transforms
        self.model_transforms = transforms.Compose([
            EnsureExactChannels(num_channels=3, num_spatial_dimensions=num_spatial_dimensions),
            transforms.ScaleIntensityRange(a_min=min_val, a_max=max_val, b_min=-1.0, b_max=1.0),
            transforms.ToTensor(),
        ])

    def _compute_metric(self, predictions, ground_truths):
        # Preprocess data as expected by VGG
        transformed_predictions = torch.stack([self.model_transforms(prediction) for prediction in predictions], dim=0)
        transformed_ground_truths = torch.stack([self.model_transforms(ground_truth) for ground_truth in ground_truths], dim=0)

        # Convert 3D data to 2D
        transformed_predictions_2d = transformed_predictions.permute(-5, -3, -4, -2, -1).contiguous().view(-1, 3, *transformed_predictions.shape[-2:]) if transformed_predictions.dim() == 5 else transformed_predictions
        transformed_ground_truths_2d = transformed_ground_truths.permute(-5, -3, -4, -2, -1).contiguous().view(-1, 3, *transformed_ground_truths.shape[-2:]) if transformed_ground_truths.dim() == 5 else transformed_ground_truths

        # Compute LPIPS score
        return torch.mean(self.criterion(transformed_predictions_2d, transformed_ground_truths_2d)).reshape(1, 1)

    def to(self, *args, **kwargs):
        # Required to move attributes to device, change their dtype, and more
        self.criterion.to(*args, **kwargs)

        return self

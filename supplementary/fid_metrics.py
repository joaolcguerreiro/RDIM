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
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
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

        for metric in self.metrics:
            # Compute metric scores
            if metric.get('dataset_level', False):
                metric['func'].update(predictions=predictions, ground_truths=ground_truths)
            else:
                scores = metric['func'](y_pred=predictions, y=ground_truths).squeeze(dim=-1).tolist()

                # Store metric score
                metric['scores'] += scores

        return {metric['name']: np.nan if metric.get('dataset_level', False) else np.mean(metric['scores'][-len(scores):]) for metric in self.metrics}

    def get_scores(self):
        return {metric['name']: [metric['func'].compute()] if metric.get('dataset_level', False) else metric['scores'] for metric in self.metrics}

    def summary(self):
        print_divisor()

        print(tabulate([
            [metric['name'], metric['func'].compute() if metric.get('dataset_level', False) else np.mean(metric['scores'])] for metric in self.metrics
        ], headers=['Metric', 'Score']))

        print_divisor()

    def preprocess(self, predictions, ground_truths):
        """
        Method to be optionally overridden by subclasses for preprocessing.
        Default implementation returns the inputs unchanged.
        """

        return predictions, ground_truths

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
            {'name': 'FID', 'scores': [], 'func': FIDMetric(num_spatial_dimensions=num_spatial_dimensions, min_val=0.0, max_val=1.0), 'dataset_level': True}
        ]

    def preprocess(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions])
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths])

        return predictions, ground_truths


class FIDMetric:
    """
    Compute Frechet Inception Distance (FID) between two sets of tensors using the pytorch-fid implementation.

    Expected inputs are tensors in [0, 1].
    For 3D data, volumes are converted to 2D slices before computing Inception activations.
    """

    def __init__(self, num_spatial_dimensions: int, min_val: float, max_val: float, dims: int = 2048) -> None:
        self.num_spatial_dimensions = num_spatial_dimensions
        self.min_val = min_val
        self.max_val = max_val
        self.dims = dims

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx])
        self.model.eval()

        self.fake_activations = []
        self.real_activations = []

        # Network transforms
        self.model_transforms = transforms.Compose([
            EnsureExactChannels(num_channels=3, num_spatial_dimensions=num_spatial_dimensions),
            transforms.ScaleIntensityRange(a_min=min_val, a_max=max_val, b_min=0.0, b_max=1.0),
            transforms.ToTensor(),
        ])

    def _prepare_images(self, images):
        # Preprocess data as expected by InceptionV3
        images = torch.stack([self.model_transforms(image) for image in images], dim=0)

        # Convert 3D data to 2D
        images = images.permute(-5, -3, -4, -2, -1).contiguous().view(-1, 3, *images.shape[-2:]) if images.dim() == 5 else images

        return images.float().clamp(0.0, 1.0)

    @torch.no_grad()
    def _get_activations(self, images):
        device = next(self.model.parameters()).device

        images = images.to(device)
        activations = self.model(images)[0]

        if activations.size(2) != 1 or activations.size(3) != 1:
            activations = torch.nn.functional.adaptive_avg_pool2d(activations, output_size=(1, 1))

        return activations.squeeze(-1).squeeze(-1).detach().cpu().numpy()

    @torch.no_grad()
    def update(self, predictions, ground_truths):
        fake_images = self._prepare_images(predictions)
        real_images = self._prepare_images(ground_truths)

        self.fake_activations.append(self._get_activations(fake_images))
        self.real_activations.append(self._get_activations(real_images))

    def compute(self):
        if len(self.fake_activations) == 0 or len(self.real_activations) == 0:
            return np.nan

        fake_activations = np.concatenate(self.fake_activations, axis=0)
        real_activations = np.concatenate(self.real_activations, axis=0)

        if fake_activations.shape[0] < 2 or real_activations.shape[0] < 2:
            return np.nan

        fake_mu = np.mean(fake_activations, axis=0)
        fake_sigma = np.cov(fake_activations, rowvar=False)

        real_mu = np.mean(real_activations, axis=0)
        real_sigma = np.cov(real_activations, rowvar=False)

        return float(calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma))

    def to(self, *args, **kwargs):
        # Required to move attributes to device, change their dtype, and more
        self.model.to(*args, **kwargs)

        return self


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

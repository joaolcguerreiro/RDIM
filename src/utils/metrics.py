import os
import sys
import torch
import numpy as np
from monai.metrics import PSNRMetric, SSIMMetric
from tabulate import tabulate
sys.path.append(os.path.join('..', '..'))
from src.utils.logger import print_divisor
from src.utils.utils import denormalize, min_max_scaling


class Metrics:
    def __init__(self):
        super().__init__()

        self.metrics = []

    def compute_scores(self, predictions, ground_truths):
        # Preprocessing
        predictions, ground_truths = self.preprocess(predictions=predictions, ground_truths=ground_truths)

        for metric in self.metrics:
            # Compute metric scores
            scores = metric['func'](y_pred=predictions, y=ground_truths).to('cpu').squeeze(dim=-1).tolist()

            # Store metric score
            metric['scores'] += scores

        return {metric['name']: np.mean(metric['scores'][-len(scores)]) for metric in self.metrics}

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


class SuperResolutionMetrics(Metrics):
    def __init__(self, num_spatial_dims, _min, _max, mean=None, std=None):
        super().__init__()

        # Define statistics
        self._min = _min
        self._max = _max
        self.mean = mean
        self.std = std

        # Define metrics
        self.metrics = [
            {'name': 'PSNR', 'scores': [], 'func': PSNRMetric(max_val=1.0)},
            {'name': 'SSIM', 'scores': [], 'func': SSIMMetric(spatial_dims=num_spatial_dims, data_range=1.0)}
        ]

    def preprocess(self, predictions, ground_truths):
        # Denormalize and scale
        predictions = torch.stack([min_max_scaling(denormalize(prediction, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for prediction in predictions])
        ground_truths = torch.stack([min_max_scaling(denormalize(ground_truth, mean=self.mean, std=self.std), _min=self._min, _max=self._max) for ground_truth in ground_truths])

        return predictions, ground_truths

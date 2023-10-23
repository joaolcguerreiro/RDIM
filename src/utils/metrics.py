import os
import sys
import numpy as np
from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric
from tabulate import tabulate
sys.path.append(os.path.join('..', '..'))
from src.utils.logger import print_divisor


class Metrics:
    def __init__(self):
        super().__init__()

        self.metrics = []

    def compute_scores(self, prediction, ground_truth):
        for metric in self.metrics:
            # Compute metric score
            score = np.mean(metric['func'](y_pred=prediction, y=ground_truth).to('cpu').numpy())

            # Store metric score
            metric['scores'].append(score)

        return {metric['name']: metric['scores'][-1] for metric in self.metrics}

    def get(self):
        return {metric['name']: metric['scores'] for metric in self.metrics}

    def summary(self):
        print_divisor()

        print(tabulate([
            [metric['name'], np.mean(metric['scores'])] for metric in self.metrics
        ], headers=['Metric', 'Score']))

        print_divisor(last=True)


class SuperResolutionMetrics(Metrics):
    def __init__(self, _max, data_range, num_spatial_dims):
        super().__init__()

        # Define metrics
        self.metrics = [
            {'name': 'MSE', 'scores': [], 'func': MSEMetric()},
            {'name': 'PSNR', 'scores': [], 'func': PSNRMetric(max_val=_max)},
            {'name': 'SSIM', 'scores': [], 'func': SSIMMetric(spatial_dims=num_spatial_dims, data_range=data_range)}
        ]

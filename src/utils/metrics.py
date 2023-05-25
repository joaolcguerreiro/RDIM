import numpy as np
from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric
from tabulate import tabulate


class Metrics:
    def __init__(self):
        super().__init__()

        self.metrics = []

    def compute_scores(self, prediction, ground_truth):
        for metric in self.metrics:
            metric['scores'].append(metric['func'](y_pred=prediction, y=ground_truth))

        return {metric['name']: metric['scores'][-1] for metric in self.metrics}

    def get(self):
        return {metric['name']: metric['scores'] for metric in self.metrics}

    def summary(self):
        print("=" * 115)

        print(tabulate([
            [metric['name'], np.mean(metric['scores'])] for metric in self.metrics
        ], headers=['Metric', 'Score']))

        print("=" * 115)


class SuperResolutionMetrics(Metrics):
    def __init__(self):
        super().__init__()

        # Define metrics
        self.metrics = [
            {'name': 'MSE', 'scores': [], 'func': MSEMetric()},
            {'name': 'PSNR', 'scores': [], 'func': PSNRMetric()},
            {'name': 'SSIM', 'scores': [], 'func': SSIMMetric()}
        ]

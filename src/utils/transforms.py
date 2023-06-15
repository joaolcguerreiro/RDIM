import torch
import numpy as np
import monai.transforms as transforms
from typing import Any, Optional, Tuple, Sequence, Union
from monai.config import DtypeLike
from monai.data.meta_obj import get_track_meta
from monai.utils.misc import ensure_tuple_size
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor


class RandGaussianSmooth(RandomizableTransform):
    """
    Apply Gaussian smooth to the input data based on randomly selected `sigma` parameters.

    Args:
        sigma: randomly select sigma value for the first spatial dimension.
        prob: probability of Gaussian smooth.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, prob: float = 0.1, sigma: Tuple[float, float] = (0.25, 1.5), approx: str = "erf") -> None:
        RandomizableTransform.__init__(self, prob)
        self.sigma = sigma
        self.rand_sigma = self.sigma[0]
        self.approx = approx

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        self.rand_sigma = self.R.uniform(low=self.sigma[0], high=self.sigma[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())

        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        sigma = ensure_tuple_size(tup=(self.rand_sigma, self.rand_sigma, self.rand_sigma), dim=img.ndim - 1)

        return transforms.GaussianSmooth(sigma=sigma, approx=self.approx)(img)


class RandForegroundGaussianNoise(RandomizableTransform):
    """
    Add Gaussian noise to volume.

    Args:
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, prob: float = 0.1, mean: float = 0.0, std: float = 0.1, dtype: DtypeLike = np.float32) -> None:
        RandomizableTransform.__init__(self, prob)
        self.mean = mean
        self.std = std
        self.dtype = dtype
        self.noise: Optional[np.ndarray] = None
        self.min = None
        self.max = None

    def randomize(self, volume: NdarrayOrTensor, mean: Optional[float] = None) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        # Get the volume min and max
        self.min = torch.min(volume)
        self.max = torch.max(volume)
        volume_range = self.max - self.min

        # Scale the mean and standard deviation by the range of the volume
        scaled_mean = (self.mean if mean is None else mean) * volume_range
        scaled_std = self.std * volume_range

        rand_std = self.R.uniform(0, scaled_std)
        noise = self.R.normal(scaled_mean, rand_std, size=volume.shape)
        self.noise, *_ = convert_data_type(noise, dtype=self.dtype)

    def __call__(self, volume: NdarrayOrTensor, mean: Optional[float] = None, randomize: bool = True) -> NdarrayOrTensor:
        # Calculate foreground and background masks
        foreground_mask = transforms.ForegroundMask(invert=True)(volume)
        background_mask = torch.logical_not(foreground_mask)

        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if randomize:
            self.randomize(volume=volume, mean=self.mean if mean is None else mean)

        if not self._do_transform:
            return volume

        if self.noise is None:
            raise RuntimeError("please call the `randomize()` function first.")

        volume, *_ = convert_data_type(volume, dtype=self.dtype)
        noise, *_ = convert_to_dst_type(self.noise, volume)

        # Calculate the noisy foreground
        noisy_foreground = (volume + noise) * foreground_mask

        # Join noisy foreground with original background
        volume = volume * background_mask + noisy_foreground

        # Clip values to the min and max of the original volume to keep the same range
        volume = torch.clamp(volume, self.min, self.max)

        return volume


class RandForegroundStreakNoise(RandomizableTransform):
    """
    Add random streaks to volume from bottom corners and middle top.
    Streaks are simulated as lines with random slope and intensity.

    Args:
        prob (float): Probability to add streak noise.
        intensity (float): The intensity of the streaks.
        line_thickness (Tuple[int, int, int]): The thickness of the lines that form the streaks, one per origin.
        num_streaks (Tuple[int, int, int]): The maximum number of streaks to generate from each origin.
        origins (Tuple[bool, bool, bool]): Flags to specify whether to generate streaks from each of the three origins: left-bottom corner, right-bottom corner, top-middle.
        dtype (DtypeLike): Output data type, if None, same as input image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, prob: float = 0.1, intensity: float = 0.1, line_thickness: Tuple[int, int, int] = (3, 3, 3), num_streaks: Tuple[int, int, int] = (10, 10, 10), origins: Tuple[bool, bool, bool] = (True, True, True), dtype: DtypeLike = np.float32) -> None:
        RandomizableTransform.__init__(self, prob)
        self.intensity = intensity
        self.line_thickness = line_thickness
        self.num_streaks = num_streaks
        self.origins = origins
        self.dtype = dtype
        self.streak: Optional[np.ndarray] = None
        self.min = None
        self.max = None

    def randomize(self, volume: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return

        # Get the volume min and max
        self.min = torch.min(volume)
        self.max = torch.max(volume)
        volume_range = self.max - self.min

        # Create an empty array for streaks
        self.streak = np.zeros_like(volume, dtype=self.dtype)

        # Random number of streaks
        num_streaks = (self.R.randint(0, self.num_streaks[0]), self.R.randint(0, self.num_streaks[1]), self.R.randint(0, self.num_streaks[2]))

        if self.origins[0]:
            for _ in range(num_streaks[0]):
                # Random intensity of the streak within the volume range
                intensity = self.R.uniform(0, self.intensity * volume_range)

                # Random slope for the line, create two for each corner and one for the top middle
                slope_lb = self.R.uniform(-1, 1)

                # Create streaks for left bottom
                x_lb = np.arange(volume.shape[-1])
                y_lb = (slope_lb * (x_lb - volume.shape[-2] * 2) + volume.shape[-2]).astype(np.int64)

                # Adjust line thickness and intensity
                line_thickness = self.R.randint(low=0, high=self.line_thickness[0] + 1)

                for i in range(-line_thickness + 1, line_thickness):
                    self.streak[..., np.clip(y_lb + i, 0, volume.shape[-2] - 1), np.clip(volume.shape[-1] - 1 - x_lb, 0, volume.shape[-1] - 1)] += intensity / (np.abs(i) * (3 / 2) + 5)

        if self.origins[1]:
            for _ in range(num_streaks[1]):
                # Random intensity of the streak within the volume range
                intensity = self.R.uniform(0, self.intensity * volume_range)

                # Random slope for the line, create two for each corner and one for the top middle
                slope_rb = self.R.uniform(-1, 1)

                # Create streaks right bottom
                x_rb = np.arange(volume.shape[-1])
                y_rb = (slope_rb * (x_rb - volume.shape[-2] * 2) + volume.shape[-2]).astype(np.int64)

                # Adjust line thickness and intensity
                line_thickness = self.R.randint(low=0, high=self.line_thickness[1] + 1)

                for i in range(-line_thickness + 1, line_thickness):
                    self.streak[..., np.clip(y_rb + i, 0, volume.shape[-2] - 1), np.clip(x_rb, 0, volume.shape[-1] - 1)] += intensity / (np.abs(i) * (3 / 2) + 5)

        if self.origins[2]:
            for _ in range(num_streaks[2]):
                # Random intensity of the streak within the volume range
                intensity = self.R.uniform(0, self.intensity * volume_range)

                # Random slope for the line, create two for each corner and one for the top middle
                slope_tm = self.R.uniform(-1, 1)

                # Create streaks for top middle
                y_tm = np.arange(volume.shape[-2])
                x_tm = (slope_tm * (y_tm + volume.shape[-2] // 4) + volume.shape[-1] // 2).astype(np.int64)

                # Adjust line thickness and intensity
                line_thickness = self.R.randint(low=0, high=self.line_thickness[2] + 1)

                for i in range(-line_thickness + 1, line_thickness):
                    self.streak[..., np.clip(y_tm + i, 0, volume.shape[-2] - 1), np.clip(x_tm, 0, volume.shape[-1] - 1)] += intensity / (np.abs(i) * (3 / 2) + 5)

        # Convert streak to desired type
        self.streak, *_ = convert_data_type(self.streak, dtype=self.dtype)

    def __call__(self, volume: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        # Calculate foreground and background masks
        foreground_mask = transforms.ForegroundMask(invert=True)(volume)
        background_mask = torch.logical_not(foreground_mask)

        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if randomize:
            self.randomize(volume=volume)

        if not self._do_transform:
            return volume

        if self.streak is None:
            raise RuntimeError("please call the `randomize()` function first.")

        volume, *_ = convert_data_type(volume, dtype=self.dtype)
        streak, *_ = convert_data_type(self.streak, dtype=self.dtype)

        # Add the streak to the volume
        volume = torch.tensor(volume + streak) * foreground_mask + volume * background_mask

        # Clip values to the min and max of the original volume to keep the same range
        volume = torch.clamp(volume, self.min, self.max)

        return volume


class NormalizeIntensity(Transform):
    """
    Normalize the input image by subtracting the mean and dividing by the standard deviation.
    Supports both 2D and 3D images, grayscale and RGB channels.

    Args:
        mean: mean value(s) for normalization. If grayscale, provide a single float value. If RGB, provide a sequence of three floats.
        std: standard deviation value(s) for normalization. Same format as `mean`.
        channel_wise: if True, normalize each channel separately. Please ensure that the first dimension represents the channel of the image if True.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mean: Union[Sequence[float], float],
        std: Union[Sequence[float], float],
        channel_wise: bool = False,
    ) -> None:
        self.mean = mean
        self.std = std
        self.channel_wise = channel_wise

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        normalized_img = convert_to_tensor(img, track_meta=False)
        ret: NdarrayOrTensor

        if self.channel_wise:
            if isinstance(self.mean, float):
                mean = [self.mean] * normalized_img.shape[0]
            else:
                mean = self.mean

            if isinstance(self.std, float):
                std = [self.std] * normalized_img.shape[0]
            else:
                std = self.std

            normalized_img = self._channel_wise_normalize(normalized_img, mean, std)
            ret = normalized_img
        else:
            mean = torch.tensor(self.mean)
            std = torch.tensor(self.std)
            normalized_img = self._normalize(normalized_img, mean, std)
            ret = normalized_img

        ret = convert_to_dst_type(ret, dst=img, dtype=normalized_img.dtype)[0]

        return ret

    def _normalize(self, normalized_img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        normalized_img = (normalized_img - mean) / std

        return normalized_img

    def _channel_wise_normalize(
        self, normalized_img: torch.Tensor, mean: Sequence[float], std: Sequence[float]
    ) -> torch.Tensor:
        for i in range(normalized_img.shape[0]):
            normalized_img[i] = (normalized_img[i] - mean[i]) / std[i]

        return normalized_img

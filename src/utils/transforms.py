import torch
import numpy as np
import monai.transforms as transforms
from typing import Any, Optional, Tuple, Sequence, Union
from monai.config import DtypeLike
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.meta_obj import get_track_meta
from monai.utils.misc import ensure_tuple_size
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform, RandomizableTransform
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

        sigma = ensure_tuple_size(vals=(self.rand_sigma, self.rand_sigma, self.rand_sigma), dim=img.ndim - 1)

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
    Add random streaks to a volume from a set of origins along a circular path, using Poisson noise.

    Args:
        prob (float): Probability to add streak noise.
        intensity (float): The intensity of the streaks.
        line_thickness (int): The thickness of the lines that form the streaks.
        num_origins (tuple[int, int] | int): The number or range of origins along the circular path.
        num_streaks (tuple[int, int] | int): The number or range of streaks to generate from each origin.
        origin_radius_ratio (float, optional): The ratio to scale the circular path radius for the origins.
                                                If it's greater than 1, the origins will be outside the volume.
                                                If it's less than 1, the origins will be inside the volume.
                                                Default is 1 (no scaling, origins lie on the volume boundary).
        slope_variation (float): Control streak slope variation.
        white_noise (bool): Define streaks color range.
                                                If true streaks are brighter.
                                                If false streaks both dark and bright (span grayscale range).
        dtype (DtypeLike): Output data type, if None, same as input image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, prob: float = 0.1, intensity: float = 0.1, line_thickness: int = 3, num_origins: tuple[int, int] | int = 360, num_streaks: tuple[int, int] | int = 10, origin_radius_ratio: float = 1.5, slope_variation: float = 1.0,  white_noise: bool = True, dtype: DtypeLike = np.float32) -> None:
        RandomizableTransform.__init__(self, prob)
        self.intensity = intensity
        self.line_thickness = line_thickness
        self.num_origins = num_origins
        self.num_streaks = num_streaks
        self.origin_radius_ratio = origin_radius_ratio
        self.slope_variation = slope_variation
        self.white_noise = white_noise
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

        # Center of the volume
        center_y, center_x = volume.shape[-2] // 2, volume.shape[-1] // 2

        rand_num_origins = self.R.randint(0, self.num_origins) if isinstance(self.num_origins, int) else self.R.randint(self.num_origins[0], self.num_origins[1])

        for origin in range(rand_num_origins):
            # angle for current origin
            theta = 2 * np.pi * origin / rand_num_origins

            rand_num_streaks = self.R.randint(0, self.num_streaks) if isinstance(self.num_streaks, int) else self.R.randint(self.num_streaks[0], self.num_streaks[1])

            for _ in range(rand_num_streaks):
                # Random intensity of the streak within the volume range
                intensity = self.R.uniform(0 if self.white_noise else -volume_range, volume_range) * self.intensity / 50

                # Generate streaks along the line y = slope * x
                slope = self.R.uniform(-self.slope_variation, self.slope_variation)
                x_line = np.arange(-center_x * self.origin_radius_ratio, volume.shape[-1] * self.origin_radius_ratio - center_x)
                y_line = (slope * x_line).astype(int)

                # Reduce intensity from streaks that are far from the volume center
                intensity = intensity / 2 if np.abs(slope) > 0.2 else intensity

                # Rotate the line by the angle theta and shift the origin to the edge of the circle
                radius = min(center_x, center_y) * self.origin_radius_ratio
                origin_x = int(center_x + radius * np.cos(theta))
                origin_y = int(center_y + radius * np.sin(theta))
                x = origin_x + x_line * np.cos(theta) - y_line * np.sin(theta)
                y = origin_y + x_line * np.sin(theta) + y_line * np.cos(theta)

                # Adjust line thickness and intensity
                line_thickness = self.R.randint(low=0, high=self.line_thickness + 1)

                for i in range(-line_thickness + 1, line_thickness):
                    # Generate streaks with varying intensity and thickness
                    x_coords = np.clip(x + i, 0, volume.shape[-1] - 1).astype(int)
                    y_coords = np.clip(y + i, 0, volume.shape[-2] - 1).astype(int)
                    self.streak[..., y_coords, x_coords] += intensity / (np.abs(i) * (3 / 2) + 1)

        # Convert streak to desired type
        self.streak, *_ = convert_data_type(np.clip(self.streak, self.min, self.max), dtype=self.dtype)

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
        volume = torch.tensor(self.R.poisson(volume - self.min + streak - streak.min()) + self.min + streak.min()) * foreground_mask + volume * background_mask

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


class RandSpatialCropPair(RandomizableTransform):
    """
    Randomly crop a pair of low-resolution (LR) and high-resolution (HR) volumes with the same spatial coordinates.

    Args:
        lr_roi_size: Low resolution region of interest size for the crop.
        scale_factor: Scale factor between HR and LR volumes.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, lr_roi_size: Union[Sequence[int], int], scale_factor: int = 2) -> None:
        RandomizableTransform.__init__(self, 1.0)
        self.lr_roi_size = ensure_tuple(lr_roi_size)
        self.scale_factor = scale_factor
        self._start_pos = (0,) * len(self.lr_roi_size)

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        # Exclude channel dimension
        spatial_dims = len(data.shape) - 1

        # Ensure lr_roi_size matches with spatial dimensions
        if len(self.lr_roi_size) < spatial_dims:
            self.lr_roi_size = ensure_tuple_rep(self.lr_roi_size, spatial_dims)

        start_pos = []
        for i in range(spatial_dims):
            if self.lr_roi_size[i] > 0:
                max_val = data.shape[-spatial_dims + i] - self.lr_roi_size[i] + 1
                start_pos.append(self.R.randint(max_val))
            else:
                # Start from the beginning for negative ROI sizes
                start_pos.append(0)

        self._start_pos = tuple(start_pos)

    def __call__(self, lr_volume: torch.Tensor, hr_volume: torch.Tensor, randomize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if randomize:
            self.randomize(data=lr_volume)

        if not self._do_transform:
            return lr_volume, hr_volume

        # Crop LR
        lr_slicer = [slice(None)] + [slice(start, start + size) if size > 0 else slice(None) for start, size in zip(self._start_pos, self.lr_roi_size)]
        lr_volume_crop = lr_volume[lr_slicer]

        # Scale start coordinates and size for HR
        hr_start_pos = tuple(pos*self.scale_factor for pos in self._start_pos)
        hr_roi_size = tuple(size*self.scale_factor if size > 0 else size for size in self.lr_roi_size)

        # Crop HR
        hr_slicer = [slice(None)] + [slice(start, start + size) if size > 0 else slice(None) for start, size in zip(hr_start_pos, hr_roi_size)]
        hr_volume_crop = hr_volume[hr_slicer]

        return lr_volume_crop, hr_volume_crop

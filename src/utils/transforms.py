import torch
import warnings
import numpy as np
import monai.transforms as transforms
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Sequence, Union
from monai.config import DtypeLike
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.meta_obj import get_track_meta
from monai.utils.misc import ensure_tuple_size
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform, RandomizableTransform
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor
from monai.transforms.utils_pytorch_numpy_unification import where


class EnsureChannelExists(Transform):
    """
    Adds a channel dimension to a 3D input tensor if it does not exist.

    Args:
        strict_check: whether to raise an error when the input is not a 3D or 4D tensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, strict_check: bool = True):
        self.strict_check = strict_check

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if len(volume.shape) not in [3, 4]:
            msg = "Input tensor is not 3D or 4D. EnsureChannelExists may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return volume

        # Add the channel dimension if it does not exist
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(dim=0)

        return volume


class EnsureExactChannels(Transform):
    """
    Ensures a 4D input tensor has exactly `num_channels` channels. If it does not, it either repeats or reshapes the channels.

    Args:
        num_channels: the exact number of channels the input tensor should have after applying the transform.
        strict_check: whether to raise an error when the input is not a 4D tensor or flexibility is not intended while the input size is invalid for an integral reshape.
        flexible_reshape: whether to allow flexible reshaping when the number of channels is not a multiple or divisor of the current number of channels in the volume.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, num_channels: int, flexible_reshape: bool = True, strict_check: bool = True):
        self.num_channels = num_channels
        self.strict_check = strict_check
        self.flexible_reshape = flexible_reshape

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if len(volume.shape) != 4:
            msg = "Input tensor is not 4D. EnsureExactChannels may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return volume

        # Get number of channels
        num_volume_channels = volume.shape[0]

        if num_volume_channels < self.num_channels:
            if self.num_channels % num_volume_channels != 0 and not self.flexible_reshape:
                msg = f"The number of channels ({self.num_channels}) must be a multiple of the current number of channels in the volume ({num_volume_channels}). EnsureExactChannels may not be applied correctly."

                if self.strict_check:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return volume

            repeats = (self.num_channels + num_volume_channels - 1) // num_volume_channels
            volume = volume.repeat(repeats, 1, 1, 1)
            volume = volume[:self.num_channels, :, :, :]

        elif num_volume_channels > self.num_channels:
            if num_volume_channels % self.num_channels != 0 and not self.flexible_reshape:
                msg = f"The number of channels ({self.num_channels}) must be a divisor of the current number of channels in the volume ({num_volume_channels}). EnsureExactChannels may not be applied correctly."

                if self.strict_check:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return volume

            # Flatten the tensor
            flattened_volume = volume.view(-1)

            # Compute target elements excluding last dimension
            target_elements = self.num_channels * volume.shape[-3] * volume.shape[-2]

            # Compute the number of elements in the last dimension based on total elements
            num_last_dim_elements = len(flattened_volume) // target_elements

            # Trim the excess and reshape to the desired shape
            volume = flattened_volume[:self.num_channels * volume.shape[-3] * volume.shape[-2] * num_last_dim_elements].view(
                self.num_channels, volume.shape[-3], volume.shape[-2], num_last_dim_elements
            )

        return volume


class NormalizeIntensity(Transform):
    """
    Normalize the input image by subtracting the mean and dividing by the standard deviation.

    Args:
        mean: mean value(s) for normalization. If grayscale, provide a single float value. If RGB, provide a sequence of three floats.
        std: standard deviation value(s) for normalization. Same format as `mean`.
        channel_wise: if True, normalize each channel separately. Please ensure that the first dimension represents the channel of the image if True.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mean: Union[float, Sequence[float], NdarrayOrTensor],
        std: Union[float, Sequence[float], NdarrayOrTensor],
        channel_wise: bool = False,
    ) -> None:
        self.mean = mean
        self.std = std
        self.channel_wise = channel_wise

    def _normalize(self, normalized_volume: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if not mean.numel() == 1:
            raise ValueError(f"Mean must be a singleton or scalar tensor, got {type(mean)} {mean}.")

        if not std.numel() == 1:
            raise ValueError(f"STD must be a singleton or scalar tensor, got {type(std)} {std}.")

        # Get device and dtype of the input
        device = normalized_volume.device
        dtype = normalized_volume.dtype

        # Move to device and dtype of the input
        mean = mean.to(device, dtype)
        std = std.to(device, dtype)

        # Normalize
        normalized_volume = (normalized_volume - mean) / std

        return normalized_volume

    def _channel_wise_normalize(self, normalized_volume: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if normalized_volume.shape[0] != mean.numel():
            raise ValueError(f"Mean must match number of volume channels, got {mean.numel()} instead of {normalized_volume.shape[0]}.")

        if not mean.dim() == 1 and mean.numel() > 1:
            raise ValueError(f"Mean must be one-dimensional, got {mean.dim()} instead of 1 dimension.")

        if normalized_volume.shape[0] != std.numel():
            raise ValueError(f"STD must match number of volume channels, got {std.numel()} instead of {normalized_volume.shape[0]}.")

        if not std.dim() == 1 and std.numel() > 1:
            raise ValueError(f"STD must be one-dimensional, got {std.dim()} instead of 1 dimension.")

        # Get device and dtype of the input
        device = normalized_volume.device
        dtype = normalized_volume.dtype

        # Reshape and move to same device and dtype
        mean = mean.view(-1, 1, 1, 1).to(device, dtype)
        std = std.view(-1, 1, 1, 1).to(device, dtype)

        # Normalize channel-wise
        normalized_volume = (normalized_volume - mean) / std

        return normalized_volume

    def __call__(self, volume: NdarrayOrTensor) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())
        normalized_volume = convert_to_tensor(volume, track_meta=get_track_meta())
        ret: NdarrayOrTensor

        if self.channel_wise:
            if isinstance(self.mean, (float, int, np.integer, np.floating)):
                mean = [self.mean] * normalized_volume.shape[0]
            else:
                mean = self.mean

            if isinstance(self.std, (float, int, np.integer, np.floating)):
                std = [self.std] * normalized_volume.shape[0]
            else:
                std = self.std

            mean = torch.tensor(mean)
            std = torch.tensor(std)
            normalized_volume = self._channel_wise_normalize(normalized_volume, mean, std)
            ret = normalized_volume
        else:
            mean = torch.tensor(self.mean)
            std = torch.tensor(self.std)
            normalized_volume = self._normalize(normalized_volume, mean, std)
            ret = normalized_volume

        ret = convert_to_dst_type(ret, dst=volume, dtype=normalized_volume.dtype)[0]

        return ret


class ThresholdIntensity(Transform):
    """
    Filter the intensity values of whole image to below threshold or above threshold.
    And fill the remaining parts of the image to the `cval` value.

    Args:
        threshold: the threshold to filter intensity values.
        above: filter values above the threshold or below the threshold, default is True.
        cval: value to fill the remaining parts of the image, default is 0.
        channel_wise: if True, apply the threshold to each channel separately. Default is False.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, threshold: Union[float, Sequence[float], NdarrayOrTensor], above: bool = True, cval: float = 0.0, channel_wise: bool = False) -> None:
        if isinstance(threshold, (float, int, np.integer, np.floating, NdarrayOrTensor)):
            self.threshold = threshold
        elif isinstance(threshold, Sequence) and all(isinstance(t, (float, int)) for t in threshold):
            self.threshold = threshold
        else:
            raise ValueError(
                f"Threshold must be an int, float, sequence of ints or floats, numpy array or torch tensor, got {type(threshold)} {threshold}.")

        self.above = above
        self.cval = cval
        self.channel_wise = channel_wise

    def _apply_threshold(self, volume: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        if not threshold.numel() == 1:
            raise ValueError(f"Threshold must be a singleton or scalar tensor, got {type(threshold)} {threshold}.")

        mask = volume > threshold if self.above else volume < threshold
        ret = where(mask, volume, self.cval)
        return ret

    def _channel_wise_threshold(self, volume: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        if volume.shape[0] != threshold.numel():
            raise ValueError(f"Threshold must match number of volume channels, got {threshold.numel()} instead of {volume.shape[0]}.")

        if not threshold.dim() == 1 and threshold.numel() > 1:
            raise ValueError(f"Threshold must be one-dimensional, got {threshold.dim()} instead of 1 dimension.")

        for channel in range(volume.shape[0]):
            mask = volume[channel] > threshold[channel] if self.above else volume[channel] < threshold[channel]
            volume[channel] = where(mask, volume[channel], self.cval)

        return volume

    def __call__(self, volume: NdarrayOrTensor) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if self.channel_wise:
            if isinstance(self.threshold, (float, int, np.integer, np.floating)):
                threshold = [self.threshold] * volume.shape[0]
            else:
                threshold = self.threshold

            threshold = torch.tensor(threshold)
            ret = self._channel_wise_threshold(volume, threshold)
        else:
            threshold = torch.tensor(self.threshold)
            ret = self._apply_threshold(volume, threshold)

        ret, *_ = convert_data_type(ret, dtype=volume.dtype)

        return ret


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

    def __call__(self, volume: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if randomize:
            self.randomize()

        if not self._do_transform:
            return volume

        sigma = ensure_tuple_size(vals=(self.rand_sigma, self.rand_sigma, self.rand_sigma), dim=volume.ndim - 1)

        return transforms.GaussianSmooth(sigma=sigma, approx=self.approx)(volume)


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


class RandTranslate(RandomizableTransform):
    """
    Apply a random translation to the input volume based on specified `translate_range`.

    Args:
        translate_range: The range for each dimension within which to apply translation.
                         Can be a single float, tuple of floats, or tuple of tuple of floats.
        prob: probability of translation.
        padding_mode: Padding mode for grid sampling ('zeros', 'border', or 'reflection').
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, prob: float = 0.1, translate_range: Union[float, Sequence[float], Sequence[Sequence[float]]] = 0.5, padding_mode: str = 'zeros') -> None:
        RandomizableTransform.__init__(self, prob)
        self.translate_range = translate_range
        self.rand_translate = [0, 0, 0]
        self.padding_mode = padding_mode

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        self.rand_translate = [self.R.uniform(low=low, high=high) * 2 for low, high in self.translate_range]

    def __call__(self, volume: NdarrayOrTensor) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        # Determine the number of spatial dimensions
        num_spatial_dims = len(self.rand_translate)

        # Process translate_range based on volume dimensions
        if isinstance(self.translate_range, (float, int)):
            self.translate_range = ((-self.translate_range, self.translate_range),) * num_spatial_dims
        elif isinstance(self.translate_range, (list, tuple)) and all(
                isinstance(item, (float, int)) for item in self.translate_range):
            self.translate_range = tuple((-val, val) for val in self.translate_range)
        else:
            self.translate_range = self.translate_range

        self.randomize()

        if not self._do_transform:
            return volume

        # Create an affine translation matrix
        theta = torch.eye(3, 4, device=volume.device).unsqueeze(0)
        theta[..., :3, -1] = torch.tensor(self.rand_translate, device=volume.device)

        # Add batch dimension temporarily
        volume = volume.unsqueeze(0)

        # Create an affine grid and apply the translation using grid sampling
        grid = F.affine_grid(theta, volume.size(), align_corners=False)
        volume = F.grid_sample(volume, grid, padding_mode=self.padding_mode, align_corners=False)

        return volume.squeeze(0)


class RandSpatialCrop(RandomizableTransform):
    """
    Randomly crop a volume. The ROI (Region of Interest) size is computed based on a provided low-resolution size and scale factor.

    Args:
        lr_roi_size: Low resolution region of interest size for the crop.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, lr_roi_size: Union[Sequence[int], int]) -> None:
        RandomizableTransform.__init__(self, 1.0)
        self.lr_roi_size = ensure_tuple(lr_roi_size)
        self._start_pos = (0,) * len(self.lr_roi_size)

    def randomize(self, lr_shape: Union[Sequence[int], int]) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        # Ensure lr_roi_size matches with spatial dimensions
        num_spatial_dims = len(lr_shape)

        if len(self.lr_roi_size) < num_spatial_dims:
            self.lr_roi_size = ensure_tuple_rep(self.lr_roi_size, num_spatial_dims)

        start_pos = []
        for i in range(num_spatial_dims):
            if self.lr_roi_size[i] > 0:
                max_val = lr_shape[-num_spatial_dims + i] - self.lr_roi_size[i] + 1
                start_pos.append(self.R.randint(max_val))
            else:
                # Start from the beginning for negative ROI sizes
                start_pos.append(0)

        self._start_pos = tuple(start_pos)

    def __call__(self, volume: torch.Tensor, scale_factor: int = None, randomize: bool = True) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if torch.where(torch.tensor(self.lr_roi_size) <= 0, True, False).all():
            return volume

        if scale_factor is None:
            return volume

        if randomize:
            lr_shape = tuple(dim // scale_factor for dim in volume.shape[-3:]) if scale_factor > 1 else volume.shape[-3:]
            self.randomize(lr_shape=lr_shape)

        if not self._do_transform:
            return volume

        # Scale start coordinates and size for the volume
        volume_start_pos = tuple(pos * scale_factor for pos in self._start_pos)
        volume_roi_size = tuple(size * scale_factor if size > 0 else size for size in self.lr_roi_size)

        # Crop HR
        slicer = [slice(None)] + [slice(start, start + size) if size > 0 else slice(None) for start, size in zip(volume_start_pos, volume_roi_size)]
        volume_crop = volume[slicer]

        return volume_crop

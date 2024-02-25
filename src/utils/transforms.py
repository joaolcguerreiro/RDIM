import sys
import torch
import warnings
import math
import numpy as np
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Sequence, Union
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.meta_obj import get_track_meta
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform, RandomizableTransform
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor
from skimage.transform import radon, iradon


class EnsureChannelExists(Transform):
    """
    Adds a channel dimension to a 3D input tensor if it does not exist.

    Args:
        channel_dim: Channel dimension index.
        strict_check: Whether to raise an error when the input is not a 3D or 4D tensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = 0, strict_check: bool = True):
        self.channel_dim = channel_dim
        self.strict_check = strict_check

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if volume.dim() not in [3, 4]:
            msg = "Input tensor is not 3D or 4D. EnsureChannelExists may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return volume

        # Add the channel dimension if it does not exist
        if volume.dim() == 3:
            volume = volume.unsqueeze(dim=self.channel_dim)

        return volume


class RemoveChannelDimension(Transform):
    """
    Remove singleton channel dimension to a 4D input tensor if it does exist.

    Args:
        channel_dim: Channel dimension index.
        strict_check: Whether to raise an error when the input is not a 3D or 4D tensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = 0, strict_check: bool = True):
        self.channel_dim = channel_dim
        self.strict_check = strict_check

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if volume.dim() not in [3, 4]:
            msg = "Input tensor is not 3D or 4D. EnsureChannelExists may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return volume

        # Remove the channel dimension if it does exist
        if volume.dim() == 4:
            volume = volume.squeeze(dim=self.channel_dim)

        return volume


class EnsureExactChannels(Transform):
    """
    Ensures a 4D input tensor has exactly `num_channels` channels. If it does not, it either repeats or reshapes the channels.

    Args:
        num_channels: The exact number of channels the input tensor should have after applying the transform.
        strict_check: Whether to raise an error when the input is not a 4D tensor or flexibility is not intended while the input size is invalid for an integral reshape.
        flexible_reshape: Whether to allow flexible reshaping when the number of channels is not a multiple or divisor of the current number of channels in the volume.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, num_channels: int, flexible_reshape: bool = True, strict_check: bool = True):
        self.num_channels = num_channels
        self.strict_check = strict_check
        self.flexible_reshape = flexible_reshape

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if volume.dim() != 4:
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
            volume = volume.repeat(repeats, *([1] * 3))
            volume = volume[:self.num_channels, :, :, :]

        elif num_volume_channels > self.num_channels:
            if num_volume_channels % self.num_channels != 0 and not self.flexible_reshape:
                msg = f"The number of channels ({self.num_channels}) must be a divisor of the current number of channels in the volume ({num_volume_channels}). EnsureExactChannels may not be applied correctly."

                if self.strict_check:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return volume

            # Permute tensor data and ensure original content is spatially coherent
            kernel_volume = torch.cat([volume[:((num_volume_channels // self.num_channels) * self.num_channels), ...][channel_index::(num_volume_channels // self.num_channels)] for channel_index in range(num_volume_channels // self.num_channels)], dim=-1)

            # Repeat excess channels and trim them if applicable
            excess_volume = volume[((num_volume_channels // self.num_channels) * self.num_channels):, ...]
            excess_volume = excess_volume.repeat(math.ceil(kernel_volume.shape[0] / excess_volume.shape[0]), *([1] * 3))[:kernel_volume.shape[0]] if excess_volume.numel() > 0 else None

            # Stack kernel and excess horizontally
            volume = torch.cat([kernel_volume, excess_volume], dim=-1) if excess_volume is not None else kernel_volume

        return volume


class PermuteDimensions(Transform):
    """
    Permute volume dimensions order.

    Args:
        axes: Volume dimensions order.
        strict_check: Whether to raise an error when the input is not a 3D or 4D tensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, axes: Sequence[int], strict_check: bool = True):
        self.axes = axes
        self.strict_check = strict_check

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if len(self.axes) != volume.dim():
            msg = f"Axes size {len(self.axes)} do not match number of dimensions in input tensor {volume.dim()}. PermuteDimensions may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return volume

        if not (isinstance(self.axes, Sequence) and all(isinstance(a, int) for a in self.axes)):
            raise ValueError(f"Axes must be sequence of ints, got {type(self.axes)} {self.axes}.")

        # Reorder volume dimensions
        volume = volume.permute(*self.axes).contiguous()

        return volume


class ReverseDimensions(Transform):
    """
    Reverse volume dimensions order.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self):
        pass

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        # Reverse volume dimensions
        volume = PermuteDimensions(axes=[*range(volume.dim() - 1, -1, -1)])(volume)

        return volume


class NormalizeIntensity(Transform):
    """
    Normalize the input image by subtracting the mean and dividing by the standard deviation.

    Args:
        mean: Mean value(s) for normalization. If grayscale, provide a single float value. If RGB, provide a sequence of three floats.
        std: Standard deviation value(s) for normalization. Same format as `mean`.
        channel_wise: If True, normalize each channel separately. Please ensure that the first dimension represents the channel of the image if True.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mean: Union[float, Sequence[float], NdarrayOrTensor],
        std: Union[float, Sequence[float], NdarrayOrTensor],
        channel_wise: bool = False,
    ) -> None:
        if isinstance(mean, (float, int, np.integer, np.floating, NdarrayOrTensor)):
            self.mean = mean
        elif isinstance(mean, Sequence) and all(isinstance(m, (float, int)) for m in mean):
            self.mean = mean
        else:
            raise ValueError(f"Mean must be an int, float, sequence of ints or floats, numpy array or torch tensor, got {type(mean)} {mean}.")

        if isinstance(std, (float, int, np.integer, np.floating, NdarrayOrTensor)):
            self.std = std
        elif isinstance(std, Sequence) and all(isinstance(s, (float, int)) for s in std):
            self.std = std
        else:
            raise ValueError(f"STD must be an int, float, sequence of ints or floats, numpy array or torch tensor, got {type(std)} {std}.")

        self.channel_wise = channel_wise

    def _normalize(self, volume: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if not mean.numel() == 1:
            raise ValueError(f"Mean must be a singleton or scalar tensor, got {type(mean)} {mean}.")

        if not std.numel() == 1:
            raise ValueError(f"STD must be a singleton or scalar tensor, got {type(std)} {std}.")

        # Get device and dtype of the input
        device = volume.device
        dtype = volume.dtype

        # Move to device and dtype of the input
        mean = mean.to(device, dtype)
        std = std.to(device, dtype)

        # Normalize
        volume = (volume - mean) / std

        return volume

    def _channel_wise_normalize(self, volume: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if volume.shape[0] != mean.numel():
            raise ValueError(f"Mean must match number of volume channels, got {mean.numel()} instead of {volume.shape[0]}.")

        if not mean.dim() == 1 and mean.numel() > 1:
            raise ValueError(f"Mean must be one-dimensional, got {mean.dim()} instead of 1 dimension.")

        if volume.shape[0] != std.numel():
            raise ValueError(f"STD must match number of volume channels, got {std.numel()} instead of {volume.shape[0]}.")

        if not std.dim() == 1 and std.numel() > 1:
            raise ValueError(f"STD must be one-dimensional, got {std.dim()} instead of 1 dimension.")

        # Get device and dtype of the input
        device = volume.device
        dtype = volume.dtype

        # Reshape and move to device and dtype of the input
        mean = mean.view(-1, *([1] * 3)).to(device, dtype)
        std = std.view(-1, *([1] * 3)).to(device, dtype)

        # Normalize channel-wise
        volume = (volume - mean) / std

        return volume

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
        threshold: The threshold to filter intensity values.
        above: Filter values above the threshold or below the threshold, default is True.
        cval: Value to fill the remaining parts of the image, default is 0.
        channel_wise: If True, apply the threshold to each channel separately. Default is False.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, threshold: Union[float, Sequence[float], NdarrayOrTensor], above: bool = True, cval: Union[float, Sequence[float], NdarrayOrTensor] = 0.0, channel_wise: bool = False) -> None:
        if isinstance(threshold, (float, int, np.integer, np.floating, NdarrayOrTensor)):
            self.threshold = threshold
        elif isinstance(threshold, Sequence) and all(isinstance(t, (float, int)) for t in threshold):
            self.threshold = threshold
        else:
            raise ValueError(f"Threshold must be an int, float, sequence of ints or floats, numpy array or torch tensor, got {type(threshold)} {threshold}.")

        if isinstance(cval, (float, int, np.integer, np.floating, NdarrayOrTensor)):
            self.cval = cval
        elif isinstance(cval, Sequence) and all(isinstance(c, (float, int)) for c in cval):
            self.cval = cval
        else:
            raise ValueError(f"Cval must be an int, float, sequence of ints or floats, numpy array or torch tensor, got {type(cval)} {cval}.")

        self.above = above
        self.channel_wise = channel_wise

    def _threshold(self, volume: torch.Tensor, threshold: torch.Tensor, cval: torch.Tensor) -> torch.Tensor:
        if not threshold.numel() == 1:
            raise ValueError(f"Threshold must be a singleton or scalar tensor, got {type(threshold)} {threshold}.")

        if not cval.numel() == 1:
            raise ValueError(f"Cval must be a singleton or scalar tensor, got {type(cval)} {cval}.")

        # Get device and dtype of the input
        device = volume.device
        dtype = volume.dtype

        # Move to device and dtype of the input
        threshold = threshold.to(device, dtype)
        cval = cval.to(device, dtype)

        # Clamp
        mask = volume > threshold if self.above else volume < threshold
        volume = torch.where(mask, volume, cval)

        return volume

    def _channel_wise_threshold(self, volume: torch.Tensor, threshold: torch.Tensor, cval: torch.Tensor) -> torch.Tensor:
        if volume.shape[0] != threshold.numel():
            raise ValueError(f"Threshold must match number of volume channels, got {threshold.numel()} instead of {volume.shape[0]}.")

        if not threshold.dim() == 1 and threshold.numel() > 1:
            raise ValueError(f"Threshold must be one-dimensional, got {threshold.dim()} instead of 1 dimension.")

        if volume.shape[0] != cval.numel():
            raise ValueError(f"Cval must match number of volume channels, got {cval.numel()} instead of {volume.shape[0]}.")

        if not cval.dim() == 1 and cval.numel() > 1:
            raise ValueError(f"Cval must be one-dimensional, got {cval.dim()} instead of 1 dimension.")

        # Get device and dtype of the input
        device = volume.device
        dtype = volume.dtype

        # Reshape and move to device and dtype of the input
        threshold = threshold.view(-1, *([1] * 3)).to(device, dtype)
        cval = cval.view(-1, *([1] * 3)).to(device, dtype)

        # Clamp channel-wise
        mask = volume > threshold if self.above else volume < threshold
        volume = torch.where(mask, volume, cval)

        return volume

    def __call__(self, volume: NdarrayOrTensor) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())
        clipped_volume = convert_to_tensor(volume, track_meta=get_track_meta())
        ret: NdarrayOrTensor

        if self.channel_wise:
            if isinstance(self.threshold, (float, int, np.integer, np.floating)):
                threshold = [self.threshold] * clipped_volume.shape[0]
            else:
                threshold = self.threshold

            if isinstance(self.cval, (float, int, np.integer, np.floating)):
                cval = [self.cval] * clipped_volume.shape[0]
            else:
                cval = self.cval

            threshold = torch.tensor(threshold)
            cval = torch.tensor(cval)
            clipped_volume = self._channel_wise_threshold(clipped_volume, threshold, cval)
            ret = clipped_volume
        else:
            threshold = torch.tensor(self.threshold)
            cval = torch.tensor(self.cval)
            clipped_volume = self._threshold(clipped_volume, threshold, cval)
            ret = clipped_volume

        ret = convert_to_dst_type(ret, dst=volume, dtype=clipped_volume.dtype)[0]

        return ret


class RandGaussianNoise(RandomizableTransform):
    """
    Apply Gaussian noise to the input data.

    Args:
        mean: Mean (centre) of the distribution.
        std: Standard deviation (spread) of distribution.
        scale: Whether to scale the standard deviation based on the data range.
        prob: Probability of Gaussian noise.
    """

    backend = [TransformBackends.TORCH]

    def __init__(self, prob: float = 0.1, mean: float = 0.0, std: Tuple[float, float] = (1.0, 5.0), scale: bool = False) -> None:
        RandomizableTransform.__init__(self, prob)
        self.mean = mean
        self.std = std
        self.rand_std = self.std[0]
        self.scale = scale

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        # Get the data range
        data_range = (torch.max(data) - torch.min(data)) if self.scale else 1.0

        self.rand_std = self.R.uniform(low=self.std[0] * data_range, high=self.std[1] * data_range)

    def __call__(self, volume: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if randomize:
            self.randomize(volume)

        if not self._do_transform:
            return volume

        # Add Gaussian noise
        volume = volume + self.R.normal(self.mean, self.rand_std, size=volume.shape)

        return volume


class RandTranslate(RandomizableTransform):
    """
    Apply a random translation to the input volume based on specified `translate_range`.

    Args:
        translate_range: The range for each dimension within which to apply translation.
                         Can be a single float, tuple of floats, or tuple of tuple of floats.
        padding_mode: Padding mode for grid sampling ('zeros', 'border', or 'reflection').
        prob: probability of translation.
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


class Radon(Transform):
    """
    Apply Radon transform to a 3D input tensor.

    Args:
        max_angle: Maximum angle for the Radon transform angle distribution.
        num_angles: Number of angles to use for the Radon transform.
        strict_check: Whether to raise an error when the input is not a 3D tensor.

    Note:
        The input tensor is expected to be in the form [D, H, W], where D, H, and W correspond to the Depth, Height, and Width dimensions, respectively.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, max_angle: float = 360, num_angles: int = 1080, strict_check: bool = True):
        self.max_angle = max_angle
        self.num_angles = num_angles
        self.strict_check = strict_check

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if volume.dim() != 3:
            msg = "Input tensor is not 3D. Radon may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return volume

        # Define angles for the Radon transform
        angles = np.linspace(0, self.max_angle, self.num_angles, endpoint=False)

        # Apply Radon transform for each slice in the volume
        sinograms = torch.stack([torch.tensor(radon(_slice.numpy(), theta=angles, circle=False)) for _slice in volume])

        return sinograms


class InverseRadon(Transform):
    """
    Apply Inverse Radon to a 3D input tensor representing sinograms.

    Args:
        max_angle: Maximum angle for the Radon transform angle distribution used to create the sinograms.
        num_angles: Number of angles used for the Radon transform resulting in the sinograms.
        filter_name: Name of the filter to use for Inverse Radon.
        strict_check: Whether to raise an error when the input is not a 3D tensor.

    Note:
        The input tensor is expected to be in the form [D, H, W], where D, H, and W correspond to the number of sinograms, the number of projection angles, and the number of rays per angle, respectively.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, max_angle: float = 360, num_angles: int = 1080, filter_name: str = 'ramp', strict_check: bool = True):
        self.max_angle = max_angle
        self.num_angles = num_angles
        self.filter_name = filter_name
        self.strict_check = strict_check

    def __call__(self, sinograms: torch.Tensor) -> torch.Tensor:
        sinograms = convert_to_tensor(sinograms, track_meta=get_track_meta())

        if len(sinograms.shape) != 3:
            msg = "Input tensor is not 3D. InverseRadon may not be applied correctly."

            if self.strict_check:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return sinograms

        # Define angles for the Inverse Radon transform
        angles = np.linspace(0, self.max_angle, self.num_angles, endpoint=False)

        # Apply Inverse Radon transform for each sinogram in the volume
        volume = torch.stack([torch.tensor(iradon(sinogram.numpy(), theta=angles, filter_name=self.filter_name, circle=False)) for sinogram in sinograms])

        return volume


class RandSimulateLowDose(RandomizableTransform):
    """
    Simulate Low-Dose CT from 3D input tensor in HU scale.

    Args:
        incident_flux_level: Number of X-ray photons in a specific area per unit time.
        mean: Mean (centre) of the distribution from the electronic noise.
        std: Standard deviation (spread) of distribution from the electronic noise.
        scale: Whether to scale the standard deviation based on the data range.
        clamp: Wheter to clip the output tensor with the input range.
        prob: Probability to add Gaussian noise.

    Note:
        The input tensor is expected to be in the form [D, H, W], where D, H, and W correspond to the Depth, Height, and Width dimensions, respectively.
        Voxel size units are expected to be meters.

    Usage:
        Weak noise: incident_flux_level = (1e4, 2e4) and std = (5.0, 10.0)
        Medium noise: incident_flux_level = (5e3, 1e4) and std = (5.0, 10.0)
        Strong noise: incident_flux_level = (2e3, 5e3) and std = (1.0, 2.5)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, prob: float = 0.1, incident_flux_level: Tuple[int, int] = (5e3, 1e4), mean: float = 0.0, std: Tuple[float, float] = (2.0, 5.0), scale: bool = False, clamp: bool = False) -> None:
        RandomizableTransform.__init__(self, prob)
        self.incident_flux_level = incident_flux_level
        self.rand_incident_flux_level = self.incident_flux_level[0]
        self.mean = mean
        self.std = std
        self.rand_std = self.std[0]
        self.scale = scale
        self.clamp = clamp

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if not self._do_transform:
            return None

        # Get the data range
        data_range = (torch.max(data) - torch.min(data)) if self.scale else 1.0

        self.rand_incident_flux_level = self.R.uniform(low=self.incident_flux_level[0], high=self.incident_flux_level[1])
        self.rand_std = self.R.uniform(low=self.std[0] * data_range, high=self.std[1] * data_range)

    def __call__(self, volume: NdarrayOrTensor, voxel_size: int = 1e-3, randomize: bool = True) -> NdarrayOrTensor:
        volume = convert_to_tensor(volume, track_meta=get_track_meta())

        if not self._do_transform:
            return volume

        # Define volume range
        _min = float(volume.min())
        _max = float(volume.max())

        # Define water attenuation coefficient
        u_water = 20

        # Compute linear attenuation coefficients base on water attenuation
        u_normal_dose = (u_water / 1000) * volume + u_water

        # Compute normal-dose projection data with the setup parallel beam geometry and 1080 projection angles in one rotation
        normal_dose_projection_data = Radon(max_angle=360, num_angles=1080)(u_normal_dose) * voxel_size

        # Compute the normal-dose transmission data
        normal_dose_transmission_data = torch.exp(-normal_dose_projection_data)

        if randomize:
            self.randomize(normal_dose_transmission_data)

        # Compute the low-dose transmission data
        low_dose_transmission_data = torch.tensor(self.R.poisson(self.rand_incident_flux_level * normal_dose_transmission_data) + self.R.normal(self.mean, self.rand_std, size=normal_dose_transmission_data.shape))

        # Ensure all non-negative values
        low_dose_transmission_data = torch.where(low_dose_transmission_data >= 0, low_dose_transmission_data, 0)

        # Compute low-dose projection data
        low_dose_projection_data = torch.log(self.rand_incident_flux_level / (low_dose_transmission_data + sys.float_info.epsilon))

        # Filter projection of added Poisson noise
        poisson_noise_projection_data = normal_dose_projection_data - low_dose_projection_data

        # Compute low-dose linear attenuation coefficients
        u_low_dose = u_normal_dose + InverseRadon(max_angle=360, num_angles=1080)(poisson_noise_projection_data / voxel_size)

        # Convert back to HU scale
        volume = 1000 * (u_low_dose - u_water) / u_water

        # Clamp values if applicable
        volume = volume if not self.clamp else ThresholdIntensity(threshold=_max, above=False, channel_wise=False, cval=_max)(ThresholdIntensity(threshold=_min, above=True, channel_wise=False, cval=_min)(volume))

        return volume

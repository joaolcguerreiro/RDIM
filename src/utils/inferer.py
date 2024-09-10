import warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterable
from monai.data.meta_tensor import MetaTensor
from monai.inferers import Inferer
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)
tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"


class MultiScaleConditionalSlidingWindowInferer(Inferer):
    """
    Multi-scale conditional sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    The inferer is equipped with a cache system to avoid redundant computations of conditions.
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    Args:
        roi_size: the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        scale_factor: the dimension ratio between inputs and conditions.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.
        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a tqdm progress bar.
        cache_roi_weight_map: whether to precompute the ROI weight map.
        cpu_thresh: when provided, dynamically switch to stitching on cpu (to save gpu memory)
            when input image volume is larger than this threshold (in pixels/voxels).
            Otherwise use ``"device"``. Thus, the output may end-up on either cpu or gpu.
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        with_coord: whether to pass the window coordinates to ``inputs_network``. Defaults to False.
            If True, the ``inputs_network``'s 2nd input argument should accept the window coordinates.

    Note:
        ``sw_batch_size`` denotes the max number of windows per inputs_network inference iteration,
        not the batch size of inputs.
    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        scale_factor: int = 1,
        sw_batch_size: int = 1,
        overlap: Sequence[float] | float = 0.25,
        mode: BlendMode | str = BlendMode.CONSTANT,
        sigma_scale: Sequence[float] | float = 0.125,
        padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
        cpu_thresh: int | None = None,
        buffer_steps: int | None = None,
        buffer_dim: int = -1,
        with_coord: bool = False,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.scale_factor = scale_factor
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.progress = progress
        self.cpu_thresh = cpu_thresh
        self.buffer_steps = buffer_steps
        self.buffer_dim = buffer_dim
        self.with_coord = with_coord

        # Set conditions cache
        self.conditions_cache = {}

        # compute_importance_map takes long time when computing on cpu. We thus
        # compute it once if it's static and then save it for future usage
        self.roi_weight_map = None
        try:
            if cache_roi_weight_map and isinstance(roi_size, Sequence) and min(roi_size) > 0:  # non-dynamic roi size
                if device is None:
                    device = "cpu"
                self.roi_weight_map = compute_importance_map(
                    ensure_tuple(self.roi_size), mode=mode, sigma_scale=sigma_scale, device=device
                )
            if cache_roi_weight_map and self.roi_weight_map is None:
                warnings.warn("cache_roi_weight_map=True, but cache is not created. (dynamic roi_size?)")
        except BaseException as e:
            raise RuntimeError(
                f"roi size {self.roi_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e

    @staticmethod
    def _get_scan_interval(
        image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
    ) -> tuple[int, ...]:
        """
        Compute scan interval according to the image size, roi size and overlap.
        Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
        use 1 instead to make sure sliding window works.
        """

        if len(image_size) != num_spatial_dims:
            raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
        if len(roi_size) != num_spatial_dims:
            raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

        scan_interval = []
        for i, o in zip(range(num_spatial_dims), overlap):
            if roi_size[i] == image_size[i]:
                scan_interval.append(int(roi_size[i]))
            else:
                interval = int(roi_size[i] * (1 - o))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)

    @staticmethod
    def _create_buffered_slices(slices, batch_size, sw_batch_size, buffer_dim, buffer_steps):
        """
        rearrange slices for buffering
        """

        slices_np = np.asarray(slices)
        slices_np = slices_np[np.argsort(slices_np[:, buffer_dim, 0], kind="mergesort")]
        slices = [tuple(slice(c[0], c[1]) for c in i) for i in slices_np]
        slices_np = slices_np[:, buffer_dim]

        _, _, _b_lens = np.unique(slices_np[:, 0], return_counts=True, return_index=True)
        b_ends = np.cumsum(_b_lens).tolist()  # possible buffer flush boundaries
        x = [0, *b_ends][:: min(len(b_ends), int(buffer_steps))]
        if x[-1] < b_ends[-1]:
            x.append(b_ends[-1])
        n_per_batch = len(x) - 1
        windows_range = [
            range(b * x[-1] + x[i], b * x[-1] + x[i + 1], sw_batch_size)
            for b in range(batch_size)
            for i in range(n_per_batch)
        ]
        b_slices = []
        for _s, _r in enumerate(windows_range):
            s_s = slices_np[windows_range[_s - 1].stop % len(slices) if _s > 0 else 0, 0]
            s_e = slices_np[(_r.stop - 1) % len(slices), 1]
            b_slices.append((_r.stop, s_s, s_e))  # buffer index, slice start, slice end
        windows_range = itertools.chain(*windows_range)  # type: ignore
        return slices, n_per_batch, b_slices, windows_range

    @staticmethod
    def _flatten_struct(seg_out):
        dict_keys = None
        seg_probs: tuple[torch.Tensor, ...]
        if isinstance(seg_out, torch.Tensor):
            seg_probs = (seg_out,)
        elif isinstance(seg_out, Mapping):
            dict_keys = sorted(seg_out.keys())  # track predictor's output keys
            seg_probs = tuple(seg_out[k] for k in dict_keys)
        else:
            seg_probs = ensure_tuple(seg_out)
        return dict_keys, seg_probs

    @staticmethod
    def _pack_struct(seg_out, dict_keys=None):
        if dict_keys is not None:
            return dict(zip(dict_keys, seg_out))
        if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
            return seg_out[0]
        return ensure_tuple(seg_out)

    @staticmethod
    def _compute_coords(coords, z_scale, out, patch):
        """
        sliding window batch spatial scaling indexing for multi-resolution outputs.
        """

        for original_idx, p in zip(coords, patch):
            idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
            if z_scale:
                for axis in range(2, len(idx_zm)):
                    idx_zm[axis] = slice(
                        int(original_idx[axis].start * z_scale[axis - 2]),
                        int(original_idx[axis].stop * z_scale[axis - 2])
                    )
            out[idx_zm] += p

    def clear_conditions_cache(self):
        self.conditions_cache.clear()

    def __call__(
        self,
        inputs: torch.Tensor,
        conditions: torch.Tensor,
        timesteps: torch.Tensor,
        inputs_network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        conditions_network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]] | None = None,
        cache_conditions: bool = False,
        process_fn: Callable | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Args:
            inputs: input image to be processed (assuming NCHW[D])
            conditions: condition image to be processes (assuming NCH/sW/s[D/s])
            timesteps: steps used to condition the inputs (assuming N)
            inputs_network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            conditions_network: target model to execute conditional inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            cache_conditions: whether to cache features of conditions.
            process_fn: process inference output and adjust the importance map per window
            args: optional args to be passed to ``inputs_network``.
            kwargs: optional keyword args to be passed to ``inputs_network``.
        """

        device = kwargs.pop("device", self.device)
        buffer_steps = kwargs.pop("buffer_steps", self.buffer_steps)
        buffer_dim = kwargs.pop("buffer_dim", self.buffer_dim)

        if device is None and self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in cpu memory if image is too large

        buffered = buffer_steps is not None and buffer_steps > 0
        num_spatial_dims = len(inputs.shape) - 2
        if buffered:
            if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
                raise ValueError(f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}.")
            if buffer_dim < 0:
                buffer_dim += num_spatial_dims
        overlap = ensure_tuple_rep(self.overlap, num_spatial_dims)
        for o in overlap:
            if o < 0 or o >= 1:
                raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
        compute_dtype = inputs.dtype

        # determine image spatial size and batch size
        # Note: all input images must have the same image size and batch size
        batch_size, _, *image_size_ = inputs.shape
        device = device or inputs.device
        sw_device = self.sw_device or inputs.device

        temp_meta = None
        if isinstance(inputs, MetaTensor):
            temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
        inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
        roi_size = fall_back_tuple(self.roi_size, image_size_)

        # in case that image size is smaller than roi size
        image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
        pad_size = []
        for k in range(len(inputs.shape) - 1, 1, -1):
            diff = max(roi_size[k - 2] - inputs.shape[k], 0)
            half = diff // 2
            pad_size.extend([half, diff - half])
        conditions_pad_size = []
        for idx in range(0, len(pad_size), 2):
            total_pad = pad_size[idx] + pad_size[idx + 1]
            pad_front = total_pad // (2 * self.scale_factor)
            pad_back = total_pad // self.scale_factor - pad_front
            conditions_pad_size.extend([pad_front, pad_back])
        if any(pad_size):
            inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(self.padding_mode, PytorchPadMode), value=self.cval)
            conditions = F.pad(conditions, pad=conditions_pad_size, mode=look_up_option(self.padding_mode, PytorchPadMode), value=self.cval)

        # Store all slices
        scan_interval = self._get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
        slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=not buffered)

        num_win = len(slices)  # number of windows per image
        total_slices = num_win * batch_size  # total number of windows
        windows_range: Iterable
        if not buffered:
            non_blocking = False
            windows_range = range(0, total_slices, self.sw_batch_size)
        else:
            slices, n_per_batch, b_slices, windows_range = self._create_buffered_slices(
                slices, batch_size, self.sw_batch_size, buffer_dim, buffer_steps
            )
            non_blocking, _ss = torch.cuda.is_available(), -1
            for x in b_slices[:n_per_batch]:
                if x[1] < _ss:  # detect overlapping slices
                    non_blocking = False
                    break
                _ss = x[2]

        # Create window-level importance map
        valid_patch_size = get_valid_patch_size(image_size, roi_size)
        if valid_patch_size == roi_size and (self.roi_weight_map is not None):
            importance_map_ = self.roi_weight_map
        else:
            try:
                valid_p_size = ensure_tuple(valid_patch_size)
                importance_map_ = compute_importance_map(
                    valid_p_size, mode=self.mode, sigma_scale=self.sigma_scale, device=sw_device, dtype=compute_dtype
                )
                if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                    importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
            except Exception as e:
                raise RuntimeError(
                    f"patch size {valid_p_size}, mode={self.mode}, sigma_scale={self.sigma_scale}, device={device}\n"
                    "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
                ) from e
        importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

        # stores output and count map
        output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
        # for each patch
        for slice_g in tqdm(windows_range) if self.progress else windows_range:
            slice_range = range(slice_g, min(slice_g + self.sw_batch_size, b_slices[b_s][0] if buffered else total_slices))
            unravel_slice = [
                [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
                for idx in slice_range
            ]
            conditions_unravel_slice = [
                [slice(idx // num_win, idx // num_win + 1), slice(None)] +
                [slice(int(s.start / self.scale_factor), int(s.stop / self.scale_factor)) for s in slices[idx % num_win]]
                for idx in slice_range
            ]
            time_unravel_slice = [
                [slice(idx // num_win, idx // num_win + 1)]
                for idx in slice_range
            ]
            if self.sw_batch_size > 1:
                win_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
                win_cond_data = torch.cat([conditions[win_cond_slice] for win_cond_slice in conditions_unravel_slice]).to(sw_device)
                time_data = torch.cat([timesteps[time_slice] for time_slice in time_unravel_slice]).to(sw_device)
            else:
                win_data = inputs[unravel_slice[0]].to(sw_device)
                win_cond_data = conditions[conditions_unravel_slice[0]].to(sw_device)
                time_data = timesteps[time_unravel_slice[0]].to(sw_device)
            if conditions_network:
                if cache_conditions:
                    win_cond_key = win_cond_data.cpu().detach().numpy().tobytes()

                    if win_cond_key in self.conditions_cache:
                        win_cond_data = self.conditions_cache[win_cond_key]
                    else:
                        win_cond_data = conditions_network(win_cond_data)
                        self.conditions_cache[win_cond_key] = win_cond_data
                else:
                    win_cond_data = conditions_network(win_cond_data)
            if self.with_coord:
                seg_prob_out = inputs_network(win_data, win_cond_data, time_data, unravel_slice, *args, **kwargs)  # batched patch
            else:
                seg_prob_out = inputs_network(win_data, win_cond_data, time_data, *args, **kwargs)  # batched patch

            # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
            dict_keys, seg_tuple = self._flatten_struct(seg_prob_out)
            if process_fn:
                seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
            else:
                w_t = importance_map_
            if len(w_t.shape) == num_spatial_dims:
                w_t = w_t[None, None]
            w_t = w_t.to(dtype=compute_dtype, device=sw_device)
            if buffered:
                c_start, c_end = b_slices[b_s][1:]
                if not sw_device_buffer:
                    k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                    sp_size = list(image_size)
                    sp_size[buffer_dim] = c_end - c_start
                    sw_device_buffer = [torch.zeros(size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device)]
                for p, s in zip(seg_tuple[0], unravel_slice):
                    offset = s[buffer_dim + 2].start - c_start
                    s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                    s[0] = slice(0, 1)
                    sw_device_buffer[0][s] += p * w_t
                b_i += len(unravel_slice)
                if b_i < b_slices[b_s][0]:
                    continue
            else:
                sw_device_buffer = list(seg_tuple)

            for ss in range(len(sw_device_buffer)):
                b_shape = sw_device_buffer[ss].shape
                seg_chns, seg_shape = b_shape[1], b_shape[2:]
                z_scale = None
                if not buffered and seg_shape != roi_size:
                    z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                    w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
                if len(output_image_list) <= ss:
                    output_shape = [batch_size, seg_chns]
                    output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                    # allocate memory to store the full output and the count for overlapping parts
                    new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                    output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
                    count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                    w_t_ = w_t.to(device)
                    for __s in slices:
                        if z_scale is not None:
                            __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                        count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
                if buffered:
                    o_slice = [slice(None)] * len(inputs.shape)
                    o_slice[buffer_dim + 2] = slice(c_start, c_end)
                    img_b = b_s // n_per_batch  # image batch index
                    o_slice[0] = slice(img_b, img_b + 1)
                    if non_blocking:
                        output_image_list[0][o_slice].copy_(sw_device_buffer[0], non_blocking=non_blocking)
                    else:
                        output_image_list[0][o_slice] += sw_device_buffer[0].to(device=device)
                else:
                    sw_device_buffer[ss] *= w_t
                    sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                    self._compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])
            sw_device_buffer = []
            if buffered:
                b_s += 1

        if non_blocking:
            torch.cuda.current_stream().synchronize()

        # account for any overlapping sections
        for ss in range(len(output_image_list)):
            output_image_list[ss] /= count_map_list.pop(0)

        # remove padding if image_size smaller than roi_size
        if any(pad_size):
            for ss, output_i in enumerate(output_image_list):
                zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
                final_slicing: list[slice] = []
                for sp in range(num_spatial_dims):
                    si = num_spatial_dims - sp - 1
                    slice_dim = slice(
                        int(round(pad_size[sp * 2] * zoom_scale[si])),
                        int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                    )
                    final_slicing.insert(0, slice_dim)
                output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

        final_output = self._pack_struct(output_image_list, dict_keys)
        if temp_meta is not None:
            final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
        else:
            final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

        return final_output  # type: ignore

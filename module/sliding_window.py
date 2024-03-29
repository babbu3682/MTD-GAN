# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option, optional_import

tqdm, _ = optional_import("tqdm", name="tqdm")

# __all__ = ["sliding_window_inference_multi_output"]


def sliding_window_inference_three_output(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    seg_output_image = torch.tensor(0.0, device=device)
    rec_output_image = torch.tensor(0.0, device=device)
    seg_count_map    = torch.tensor(0.0, device=device)
    rec_count_map    = torch.tensor(0.0, device=device)
    _initialized = False
    cls_list = []
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        cls_prob, seg_prob, rec_prob = predictor(window_data, *args, **kwargs)
        seg_prob = seg_prob.to(device)  # batched patch segmentation
        rec_prob = rec_prob.to(device)  # batched patch segmentation
        if window_data.sum():
            cls_list.append(cls_prob)

        if not _initialized:  # init. buffer at the first iteration
            seg_output_shape = [batch_size, seg_prob.shape[1]] + list(image_size)
            rec_output_shape = [batch_size, rec_prob.shape[1]] + list(image_size)

            # allocate memory to store the full output and the count for overlapping parts
            seg_output_image = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
            rec_output_image = torch.zeros(rec_output_shape, dtype=torch.float32, device=device)
            seg_count_map = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
            rec_count_map = torch.zeros(rec_output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            seg_output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            rec_output_image[original_idx] += importance_map * rec_prob[idx - slice_g]
            seg_count_map[original_idx] += importance_map
            rec_count_map[original_idx] += importance_map

    # account for any overlapping sections
    seg_output_image = seg_output_image / seg_count_map
    rec_output_image = rec_output_image / rec_count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(seg_output_image.shape):
        final_slicing.insert(0, slice(None))        
    return torch.concat(cls_list), seg_output_image[final_slicing], rec_output_image[final_slicing]

def sliding_window_inference_two_output_seg_rec(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    seg_output_image = torch.tensor(0.0, device=device)
    rec_output_image = torch.tensor(0.0, device=device)
    seg_count_map    = torch.tensor(0.0, device=device)
    rec_count_map    = torch.tensor(0.0, device=device)
    _initialized = False
    cls_list = []
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob, rec_prob = predictor(window_data, *args, **kwargs)
        seg_prob = seg_prob.to(device)  # batched patch segmentation
        rec_prob = rec_prob.to(device)  # batched patch segmentation

        if not _initialized:  # init. buffer at the first iteration
            seg_output_shape = [batch_size, seg_prob.shape[1]] + list(image_size)
            rec_output_shape = [batch_size, rec_prob.shape[1]] + list(image_size)

            # allocate memory to store the full output and the count for overlapping parts
            seg_output_image = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
            rec_output_image = torch.zeros(rec_output_shape, dtype=torch.float32, device=device)
            seg_count_map = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
            rec_count_map = torch.zeros(rec_output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            seg_output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            rec_output_image[original_idx] += importance_map * rec_prob[idx - slice_g]
            seg_count_map[original_idx] += importance_map
            rec_count_map[original_idx] += importance_map

    # account for any overlapping sections
    seg_output_image = seg_output_image / seg_count_map
    rec_output_image = rec_output_image / rec_count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(seg_output_image.shape):
        final_slicing.insert(0, slice(None))        
    return seg_output_image[final_slicing], rec_output_image[final_slicing]

def sliding_window_inference_two_output_cls_rec(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    rec_output_image = torch.tensor(0.0, device=device)
    rec_count_map    = torch.tensor(0.0, device=device)
    _initialized = False
    cls_list = []
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        cls_prob, rec_prob = predictor(window_data, *args, **kwargs)
        rec_prob = rec_prob.to(device)  # batched patch segmentation
        if window_data.sum():
            cls_list.append(cls_prob)

        if not _initialized:  # init. buffer at the first iteration
            rec_output_shape = [batch_size, rec_prob.shape[1]] + list(image_size)

            # allocate memory to store the full output and the count for overlapping parts
            rec_output_image = torch.zeros(rec_output_shape, dtype=torch.float32, device=device)
            rec_count_map = torch.zeros(rec_output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            rec_output_image[original_idx] += importance_map * rec_prob[idx - slice_g]
            rec_count_map[original_idx] += importance_map

    # account for any overlapping sections
    rec_output_image = rec_output_image / rec_count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(rec_output_image.shape):
        final_slicing.insert(0, slice(None))        
    return torch.concat(cls_list), rec_output_image[final_slicing]

def sliding_window_inference_two_output_cls_seg(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    seg_output_image = torch.tensor(0.0, device=device)
    seg_count_map    = torch.tensor(0.0, device=device)
    _initialized = False
    cls_list = []
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        cls_prob, seg_prob = predictor(window_data, *args, **kwargs)
        seg_prob = seg_prob.to(device)  # batched patch segmentation
        if window_data.sum():
            cls_list.append(cls_prob)

        if not _initialized:  # init. buffer at the first iteration
            seg_output_shape = [batch_size, seg_prob.shape[1]] + list(image_size)

            # allocate memory to store the full output and the count for overlapping parts
            seg_output_image = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
            seg_count_map = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            seg_output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            seg_count_map[original_idx] += importance_map

    # account for any overlapping sections
    seg_output_image = seg_output_image / seg_count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(seg_output_image.shape):
        final_slicing.insert(0, slice(None))        
    return torch.concat(cls_list), seg_output_image[final_slicing]


# def sliding_window_inference_two_output_seg_rec(
#     inputs: torch.Tensor,
#     roi_size: Union[Sequence[int], int],
#     sw_batch_size: int,
#     predictor: Callable[..., torch.Tensor],
#     overlap: float = 0.25,
#     mode: Union[BlendMode, str] = BlendMode.CONSTANT,
#     sigma_scale: Union[Sequence[float], float] = 0.125,
#     padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
#     cval: float = 0.0,
#     sw_device: Union[torch.device, str, None] = None,
#     device: Union[torch.device, str, None] = None,
#     progress: bool = False,
#     *args: Any,
#     **kwargs: Any,
# ) -> torch.Tensor:

#     num_spatial_dims = len(inputs.shape) - 2
#     if overlap < 0 or overlap >= 1:
#         raise AssertionError("overlap must be >= 0 and < 1.")

#     # determine image spatial size and batch size
#     # Note: all input images must have the same image size and batch size
#     image_size_ = list(inputs.shape[2:])
#     batch_size = inputs.shape[0]

#     if device is None:
#         device = inputs.device
#     if sw_device is None:
#         sw_device = inputs.device

#     roi_size = fall_back_tuple(roi_size, image_size_)
#     # in case that image size is smaller than roi size
#     image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
#     pad_size = []
#     for k in range(len(inputs.shape) - 1, 1, -1):
#         diff = max(roi_size[k - 2] - inputs.shape[k], 0)
#         half = diff // 2
#         pad_size.extend([half, diff - half])
#     inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

#     scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

#     # Store all slices in list
#     slices = dense_patch_slices(image_size, roi_size, scan_interval)
#     num_win = len(slices)  # number of windows per image
#     total_slices = num_win * batch_size  # total number of windows

#     # Create window-level importance map
#     importance_map = compute_importance_map(
#         get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
#     )

#     # Perform predictions
#     seg_output_image = torch.tensor(0.0, device=device)
#     seg_count_map    = torch.tensor(0.0, device=device)
#     _initialized = False

#     for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
#         slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
#         unravel_slice = [
#             [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
#             for idx in slice_range
#         ]
#         window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
#         seg_prob, _ = predictor(window_data, *args, **kwargs)
#         seg_prob = seg_prob.to(device)  # batched patch segmentation

#         if not _initialized:  # init. buffer at the first iteration
#             seg_output_shape = [batch_size, seg_prob.shape[1]] + list(image_size)

#             # allocate memory to store the full output and the count for overlapping parts
#             seg_output_image = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
#             seg_count_map = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
#             _initialized = True

#         # store the result in the proper location of the full output. Apply weights from importance map.
#         for idx, original_idx in zip(slice_range, unravel_slice):
#             seg_output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
#             seg_count_map[original_idx] += importance_map

#     # account for any overlapping sections
#     seg_output_image = seg_output_image / seg_count_map

#     final_slicing: List[slice] = []
#     for sp in range(num_spatial_dims):
#         slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
#         final_slicing.insert(0, slice_dim)
#     while len(final_slicing) < len(seg_output_image.shape):
#         final_slicing.insert(0, slice(None))        
#     return seg_output_image[final_slicing]

# def sliding_window_inference_two_output_cls_rec(
#     inputs: torch.Tensor,
#     roi_size: Union[Sequence[int], int],
#     sw_batch_size: int,
#     predictor: Callable[..., torch.Tensor],
#     overlap: float = 0.25,
#     mode: Union[BlendMode, str] = BlendMode.CONSTANT,
#     sigma_scale: Union[Sequence[float], float] = 0.125,
#     padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
#     cval: float = 0.0,
#     sw_device: Union[torch.device, str, None] = None,
#     device: Union[torch.device, str, None] = None,
#     progress: bool = False,
#     *args: Any,
#     **kwargs: Any,
# ) -> torch.Tensor:

#     num_spatial_dims = len(inputs.shape) - 2
#     if overlap < 0 or overlap >= 1:
#         raise AssertionError("overlap must be >= 0 and < 1.")

#     # determine image spatial size and batch size
#     # Note: all input images must have the same image size and batch size
#     image_size_ = list(inputs.shape[2:])
#     batch_size = inputs.shape[0]

#     if device is None:
#         device = inputs.device
#     if sw_device is None:
#         sw_device = inputs.device

#     roi_size = fall_back_tuple(roi_size, image_size_)
#     # in case that image size is smaller than roi size
#     image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
#     pad_size = []
#     for k in range(len(inputs.shape) - 1, 1, -1):
#         diff = max(roi_size[k - 2] - inputs.shape[k], 0)
#         half = diff // 2
#         pad_size.extend([half, diff - half])
#     inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

#     scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

#     # Store all slices in list
#     slices = dense_patch_slices(image_size, roi_size, scan_interval)
#     num_win = len(slices)  # number of windows per image
#     total_slices = num_win * batch_size  # total number of windows

#     # Create window-level importance map
#     importance_map = compute_importance_map(
#         get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
#     )

#     # Perform predictions
#     seg_output_image = torch.tensor(0.0, device=device)
#     seg_count_map    = torch.tensor(0.0, device=device)
#     _initialized = False
#     cls_list = []
#     for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
#         slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
#         unravel_slice = [
#             [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
#             for idx in slice_range
#         ]
#         window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
#         cls_prob, seg_prob = predictor(window_data, *args, **kwargs)
#         seg_prob = seg_prob.to(device)  # batched patch segmentation
#         if window_data.sum():
#             cls_list.append(cls_prob)

#         if not _initialized:  # init. buffer at the first iteration
#             seg_output_shape = [batch_size, seg_prob.shape[1]] + list(image_size)

#             # allocate memory to store the full output and the count for overlapping parts
#             seg_output_image = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
#             seg_count_map = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
#             _initialized = True

#         # store the result in the proper location of the full output. Apply weights from importance map.
#         for idx, original_idx in zip(slice_range, unravel_slice):
#             seg_output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
#             seg_count_map[original_idx] += importance_map

#     # account for any overlapping sections
#     seg_output_image = seg_output_image / seg_count_map

#     final_slicing: List[slice] = []
#     for sp in range(num_spatial_dims):
#         slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
#         final_slicing.insert(0, slice_dim)
#     while len(final_slicing) < len(seg_output_image.shape):
#         final_slicing.insert(0, slice(None))        
#     return torch.concat(cls_list)

# def sliding_window_inference_two_output_cls_seg(
#     inputs: torch.Tensor,
#     roi_size: Union[Sequence[int], int],
#     sw_batch_size: int,
#     predictor: Callable[..., torch.Tensor],
#     overlap: float = 0.25,
#     mode: Union[BlendMode, str] = BlendMode.CONSTANT,
#     sigma_scale: Union[Sequence[float], float] = 0.125,
#     padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
#     cval: float = 0.0,
#     sw_device: Union[torch.device, str, None] = None,
#     device: Union[torch.device, str, None] = None,
#     progress: bool = False,
#     *args: Any,
#     **kwargs: Any,
# ) -> torch.Tensor:

#     num_spatial_dims = len(inputs.shape) - 2
#     if overlap < 0 or overlap >= 1:
#         raise AssertionError("overlap must be >= 0 and < 1.")

#     # determine image spatial size and batch size
#     # Note: all input images must have the same image size and batch size
#     image_size_ = list(inputs.shape[2:])
#     batch_size = inputs.shape[0]

#     if device is None:
#         device = inputs.device
#     if sw_device is None:
#         sw_device = inputs.device

#     roi_size = fall_back_tuple(roi_size, image_size_)
#     # in case that image size is smaller than roi size
#     image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
#     pad_size = []
#     for k in range(len(inputs.shape) - 1, 1, -1):
#         diff = max(roi_size[k - 2] - inputs.shape[k], 0)
#         half = diff // 2
#         pad_size.extend([half, diff - half])
#     inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

#     scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

#     # Store all slices in list
#     slices = dense_patch_slices(image_size, roi_size, scan_interval)
#     num_win = len(slices)  # number of windows per image
#     total_slices = num_win * batch_size  # total number of windows

#     # Create window-level importance map
#     importance_map = compute_importance_map(
#         get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
#     )

#     # Perform predictions
#     seg_output_image = torch.tensor(0.0, device=device)
#     seg_count_map    = torch.tensor(0.0, device=device)
#     _initialized = False
#     cls_list = []
#     for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
#         slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
#         unravel_slice = [
#             [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
#             for idx in slice_range
#         ]
#         window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
#         cls_prob, seg_prob = predictor(window_data, *args, **kwargs)
#         seg_prob = seg_prob.to(device)  # batched patch segmentation
#         if window_data.sum():
#             cls_list.append(cls_prob)

#         if not _initialized:  # init. buffer at the first iteration
#             seg_output_shape = [batch_size, seg_prob.shape[1]] + list(image_size)

#             # allocate memory to store the full output and the count for overlapping parts
#             seg_output_image = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
#             seg_count_map = torch.zeros(seg_output_shape, dtype=torch.float32, device=device)
#             _initialized = True

#         # store the result in the proper location of the full output. Apply weights from importance map.
#         for idx, original_idx in zip(slice_range, unravel_slice):
#             seg_output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
#             seg_count_map[original_idx] += importance_map

#     # account for any overlapping sections
#     seg_output_image = seg_output_image / seg_count_map

#     final_slicing: List[slice] = []
#     for sp in range(num_spatial_dims):
#         slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
#         final_slicing.insert(0, slice_dim)
#     while len(final_slicing) < len(seg_output_image.shape):
#         final_slicing.insert(0, slice(None))        
#     return torch.concat(cls_list), seg_output_image[final_slicing]

def sliding_window_inference_cls_output(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:

    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Perform predictions
    cls_list = []
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        cls_prob = predictor(window_data, *args, **kwargs)
        if window_data.sum():
            cls_list.append(cls_prob)

    # account for any overlapping sections
    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)

    return torch.concat(cls_list)

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.
    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
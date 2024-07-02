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
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import torch
from monai.utils import StrEnum
import inspect
from packaging import version

from .scheduler import Scheduler

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True, stacklevel=2):
    from .. import __version__

    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)

    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since diffusers'"
                f" version {__version__} is >= {version_name}"
            )

        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        elif deprecated_kwargs is None:
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."

        if warning is not None:
            warning = warning + " " if standard_warn else ""
            warnings.warn(warning + message, FutureWarning, stacklevel=stacklevel)

    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(f"{function} in {filename} line {line_number-1} got an unexpected keyword argument `{key}`")

    if len(values) == 0:
        return
    elif len(values) == 1:
        return values[0]
    return values


class DPMPredictionType(StrEnum):
    """
    Set of valid prediction type names for the PNDM scheduler's `prediction_type` argument.

    epsilon: predicting the noise of the diffusion process
    v_prediction: velocity prediction, see section 2.4 https://imagen.research.google/video/paper.pdf
    """

    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"



class DPMScheduler_Multi(Scheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "linear_beta",
        prediction_type: str = DPMPredictionType.EPSILON,
        solver_order: int = 2,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        use_karras_sigmas: Optional[bool] = False,
        use_lu_lambdas: Optional[bool] = False,
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        euler_at_final: bool = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        **schedule_args,
    ) -> None:
        super().__init__(num_train_timesteps, schedule, **schedule_args)

        if prediction_type not in DPMPredictionType.__members__.values():
            raise ValueError("Argument `prediction_type` must be a member of DPMPredictionType")

        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.lower_order_final = lower_order_final
        self.lambda_min_clipped = lambda_min_clipped
        self.use_karras_sigmas = use_karras_sigmas
        self.use_lu_lambdas = use_lu_lambdas
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.variance_type = variance_type
        self.thresholding = thresholding
        self.solver_type = solver_type
        self.euler_at_final = euler_at_final
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset

        # Currently we only support VP-type noise schedule
        self.alpha_t  = torch.sqrt(self.alphas_cumprod)
        self.sigma_t  = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None

        # default the number of inference timesteps to the number of train steps
        self.set_timesteps(num_train_timesteps)

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increae 1 after each scheduler step.
        """
        return self._step_index

    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.lambda_min_clipped)
        last_timestep = ((self.num_train_timesteps - clipped_idx).numpy()).item()

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.timestep_spacing == "linspace":
            timesteps = (
                # np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
                np.linspace(0, last_timestep - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = last_timestep // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.use_karras_sigmas:
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        elif self.use_lu_lambdas:
            lambdas = np.flip(log_sigmas.copy())
            lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
            sigmas = np.exp(lambdas)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None,] * self.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_lu(self, in_lambdas: torch.FloatTensor, num_inference_steps) -> torch.FloatTensor:
        """Constructs the noise schedule of Lu et al. (2022)."""

        lambda_min: float = in_lambdas[-1].item()
        lambda_max: float = in_lambdas[0].item()

        rho = 1.0  # 1.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = lambda_min ** (1 / rho)
        max_inv_rho = lambda_max ** (1 / rho)
        lambdas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return lambdas

    def convert_model_output(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.prediction_type == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.thresholding:
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t

            return epsilon

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        noise: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        noise: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """

        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing`sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.0)) * D0
                - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        return x_t

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        index_candidates = (self.timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        self._step_index = step_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,        
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.euler_at_final or (self.lower_order_final and len(self.timesteps) < 15)
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
        else:
            noise = None

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return prev_sample, None
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        return self.num_train_timesteps
from typing import Optional, Callable

import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed, _generate_noise, \
    _get_flat_grad_sample
from torch.optim import Optimizer


class DPSGDF_Optimizer(DPOptimizer):
    """
    Customized optimizer for DPSGD-F, inherited from DPOptimizer and overwriting the following

    - clip_and_accumulate(self, per_sample_clip_bound) now takes an extra tensor list parameter indicating the clipping bound per sample
    - add_noise(self, max_grad_clip:float) takes an extra paramter ``max_grad_clip``,
        which is the maximum clipping factor among all the groups, i.e. max(per_sample_clip_bound)
    - pre_step() and step() are overwritten by taking this extra parameter
    """

    def __init__(
            self,
            optimizer: Optimizer,
            *,
            noise_multiplier: float,
            expected_batch_size: Optional[int],
            loss_reduction: str = "mean",
            generator=None,
            secure_mode: bool = False,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=0,  # not applicable for DPSGDF_Optimizer
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

    def clip_and_accumulate(self, per_sample_clip_bound):
        """
        Clips gradient according to per sample clipping bounds and accumulates gradient for a given batch
        Args:
        per_sample_clip_bound: a tensor list of clip bound per sample
        """
        # self.grad_samples are calculated from parent class, equivalent to the following
        #
        # ret = []
        # for p in self.params:
        #     ret.append(_get_flat_grad_sample(p))
        # return ret

        # For neural network, this per_param_norms is per layer's normalization across parameters, not samples
        # output dimension: num_layers * num_samples(per batch)
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]

        # torch.stack(per_param_norms, dim=1) will make the dimension num_samples * num_layers
        # per_sample_norms has dimension of num_samples
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (per_sample_clip_bound / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = _get_flat_grad_sample(p)
            # equivalent to grad = grad * min(1, Ck / norm(grad))
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self, max_grad_clip: float):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        Args:
            max_grad_clip: C = max(C_k), for all group k
        """

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * max_grad_clip,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p.grad)
            _mark_as_processed(p.summed_grad)

    def pre_step(
            self, per_sample_clip_bound: torch.Tensor, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``
        Args:
            per_sample_clip_bound: Defines the clipping bound for each sample.
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate(per_sample_clip_bound)
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise(torch.max(per_sample_clip_bound).item())
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, per_sample_clip_bound: torch.Tensor, closure: Optional[Callable[[], float]] = None) -> Optional[
        float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(per_sample_clip_bound):
            return self.original_optimizer.step()
        else:
            return None

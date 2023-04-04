from typing import Optional, Callable

import torch
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _get_flat_grad_sample, _mark_as_processed


class DPSGD_Global_Optimizer(DPOptimizer):

    def clip_and_accumulate(self, strict_max_grad_norm):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        # C = max_grad_norm
        # Z = strict_max_grad_norm
        # condition is equivalent to norm[i] <= Z
        # when condition holds, scale gradient by C/Z
        # otherwise, clip to 0
        per_sample_global_clip_factor = torch.where(per_sample_clip_factor >= self.max_grad_norm / strict_max_grad_norm,
                                                    # scale by C/Z
                                                    torch.ones_like(
                                                        per_sample_clip_factor) * self.max_grad_norm / strict_max_grad_norm,
                                                    torch.zeros_like(per_sample_clip_factor))  # clip to 0
        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)

            # refer to lines 197-199 in 
            # https://github.com/pytorch/opacus/blob/ee6867e6364781e67529664261243c16c3046b0b/opacus/per_sample_gradient_clip.py
            # as well as https://github.com/woodyx218/opacus_global_clipping README
            grad = torch.einsum("i,i...", per_sample_global_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    # note add_noise does not have to be modified since max_grad_norm = C is sensitivity

    def pre_step(
            self, strict_max_grad_norm, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``
        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.clip_and_accumulate(strict_max_grad_norm)
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, strict_max_grad_norm, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step(strict_max_grad_norm):
            return self.original_optimizer.step()
        else:
            return None

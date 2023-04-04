import torch
from opacus.optimizers.optimizer import _check_processed_flag, _get_flat_grad_sample, _mark_as_processed

from optimizers.dpsgd_global_optimizer import DPSGD_Global_Optimizer


class DPSGD_Global_Adaptive_Optimizer(DPSGD_Global_Optimizer):

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
        # otherwise, clip to norm C, note that here we remove the aggressive clipping in global method
        per_sample_global_clip_factor = torch.where(per_sample_clip_factor >= self.max_grad_norm / strict_max_grad_norm,
                                                    # scale by C/Z
                                                    torch.ones_like(
                                                        per_sample_clip_factor) * self.max_grad_norm / strict_max_grad_norm,
                                                    per_sample_clip_factor)  # clip to C
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

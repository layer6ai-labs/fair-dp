from typing import List, Union

from opacus import PrivacyEngine
from opacus.optimizers.optimizer import DPOptimizer
from optimizers.dpsgd_f_optimizer import DPSGDF_Optimizer
from torch import optim


class DPSGDF_PrivacyEngine(PrivacyEngine):
    """
    This class defines the customized privacy engine for DPSGD-F.
    Specifically, it overwrites the _prepare_optimizer() method from parent class to return DPSGDF_Optimizer
    """

    def __init__(self, *, accountant: str = "rdp", secure_mode: bool = False):
        if accountant != 'rdp':
            raise ValueError("DPSGD-F must use an RDP accountant since it composes SGM with different parameters.")

        super().__init__(accountant=accountant, secure_mode=secure_mode)

    def _prepare_optimizer(
            self,
            optimizer: optim.Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],  # not applicable in DPSGDF
            expected_batch_size: int,
            loss_reduction: str = "mean",
            distributed: bool = False,
            clipping: str = "flat",
            noise_generator=None,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return DPSGDF_Optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )

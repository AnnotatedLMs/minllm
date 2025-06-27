# Standard Library
import math


class CosineScheduler:
    """
    Cosine learning rate schedule with linear warmup.

    Schedule:
    1. Linear warmup from 0 to learning_rate over warmup_iters
    2. Cosine decay from learning_rate to min_lr over lr_decay_iters
    3. Constant min_lr after lr_decay_iters
    """

    def __init__(self, learning_rate: float, warmup_iters: int, lr_decay_iters: int, min_lr: float):
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

        # Validate
        assert min_lr <= learning_rate
        assert warmup_iters >= 0
        assert lr_decay_iters >= warmup_iters

    def _calculate_warmup_lr(self, it: int) -> float:
        """
        Calculate learning rate during warmup phase.
        Linear increase from 0 to learning_rate.
        """
        return self.learning_rate * (it + 1) / (self.warmup_iters + 1)

    def _calculate_cosine_decay_lr(self, it: int) -> float:
        """
        Calculate learning rate during cosine decay phase.
        Smoothly decreases from learning_rate to min_lr.
        """
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def _is_warmup_phase(self, it: int) -> bool:
        """Check if we're in the warmup phase."""
        return it < self.warmup_iters

    def _is_constant_phase(self, it: int) -> bool:
        """Check if we're in the constant minimum lr phase."""
        return it > self.lr_decay_iters

    def get_lr(self, it: int) -> float:
        """
        Get learning rate for iteration number.

        Args:
            it: Current iteration number (0-indexed)

        Returns:
            Learning rate for this iteration
        """
        if self._is_warmup_phase(it):
            return self._calculate_warmup_lr(it)

        if self._is_constant_phase(it):
            return self.min_lr

        # Else: We're in the cosine decay phase
        return self._calculate_cosine_decay_lr(it)

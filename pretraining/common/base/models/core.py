# Standard Library
import abc

# Third Party
import torch.nn as nn


class BaseTorchModule(nn.Module, abc.ABC):
    """
    Base class for all our modules that adds common functionality.
    """

    def get_num_params(self) -> int:
        """
        Get number of parameters in this module.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters())

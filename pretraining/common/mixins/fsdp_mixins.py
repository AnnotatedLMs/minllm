# Standard Library
import typing

# Third Party
from torch import nn

# Project
from pretraining.common.models.blocks import block_group
from pretraining.configs.training import execution_configs


class FSDPMixin:
    """
    Mixin class that provides FSDP wrapping policy generation for architectures.

    This is an abstract mixin - concrete classes must implement:
    - get_fsdp_wrappable_modules(): Module types that can be wrapped
    - get_transformer_blocks(): List of transformer blocks

    Optionally override:
    - get_fsdp_special_modules(): Special modules like embeddings/output layers
    """

    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        """
        Return module TYPES that FSDP should wrap.

        Used when: FSDP recursively walks the model and needs to decide
                   which modules to wrap based on isinstance() checks.

        Returns:
            Set of module types (classes) to wrap
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_fsdp_wrappable_modules()"
        )

    def get_transformer_blocks(self) -> typing.List[nn.Module]:
        """
        Return actual block INSTANCES for selective wrapping strategies.

        Used when: We need to wrap specific blocks (e.g., every 2nd block)
                   or group blocks together for FSDP.

        Returns:
            List of transformer block instances (in order)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_transformer_blocks()"
        )

    def get_fsdp_special_modules(
        self,
        token_embeddings: typing.Optional[nn.Module] = None,
        position_embeddings: typing.Optional[nn.Module] = None,
        lm_head: typing.Optional[nn.Module] = None,
    ) -> typing.Dict[str, nn.Module]:
        """
        Return large module INSTANCES that should be wrapped separately.

        Used when: We want to wrap embeddings/output layers separately
                   from transformer blocks for memory efficiency.

        Args:
            token_embeddings: Token embedding module
            position_embeddings: Position embedding module (optional)
            lm_head: Language model head (optional)

        Returns:
            Dict mapping module names to module instances
        """
        special_modules = {}

        if token_embeddings is not None:
            special_modules["token_embeddings"] = token_embeddings

        if position_embeddings is not None:
            special_modules["position_embeddings"] = position_embeddings

        # Only include lm_head if it's not weight-tied to token_embeddings
        if lm_head is not None and lm_head is not token_embeddings:
            special_modules["lm_head"] = lm_head

        return special_modules

    def get_fsdp_wrap_policy(
        self,
        strategy: typing.Optional[execution_configs.FSDPWrapStrategy],
    ) -> typing.Optional[typing.Callable]:
        """
        Generate an FSDP auto-wrap policy based on the strategy.

        Args:
            strategy: FSDP wrapping strategy

        Returns:
            Auto-wrap policy function or None
        """
        if strategy is None:
            return None

        if strategy == execution_configs.FSDPWrapStrategy.SIZE_BASED:
            # This will be handled by model_wrapper with size_based_auto_wrap_policy
            return None

        elif strategy == execution_configs.FSDPWrapStrategy.BY_BLOCK:
            wrappable_modules = self.get_fsdp_wrappable_modules()

            def wrap_policy(
                module: nn.Module,
                recurse: bool,
                nonwrapped_numel: int,
            ) -> bool:
                if recurse:
                    return True
                return isinstance(module, tuple(wrappable_modules))

            return wrap_policy

        elif strategy == execution_configs.FSDPWrapStrategy.BY_BLOCK_AND_SIZE:
            wrappable_modules = self.get_fsdp_wrappable_modules()
            special_modules = set(self.get_fsdp_special_modules().values())

            def wrap_policy(
                module: nn.Module,
                recurse: bool,
                nonwrapped_numel: int,
            ) -> bool:
                if recurse:
                    return True
                # Wrap transformer blocks and special modules (embeddings, output)
                return isinstance(module, tuple(wrappable_modules)) or module in special_modules

            return wrap_policy

        elif strategy in [
            execution_configs.FSDPWrapStrategy.BY_BLOCK_GROUP,
            execution_configs.FSDPWrapStrategy.BY_BLOCK_GROUP_AND_SIZE,
        ]:
            # For block groups, wrap BlockGroup instances
            def wrap_policy(
                module: nn.Module,
                recurse: bool,
                nonwrapped_numel: int,
            ) -> bool:
                if recurse:
                    return True
                # Wrap BlockGroup instances
                if isinstance(module, block_group.BlockGroup):
                    return True
                # For BY_BLOCK_GROUP_AND_SIZE, also wrap special modules
                if strategy == execution_configs.FSDPWrapStrategy.BY_BLOCK_GROUP_AND_SIZE:
                    special_modules = set(self.get_fsdp_special_modules().values())
                    return module in special_modules
                return False

            return wrap_policy

        elif strategy in [
            execution_configs.FSDPWrapStrategy.ONE_IN_TWO,
            execution_configs.FSDPWrapStrategy.ONE_IN_THREE,
            execution_configs.FSDPWrapStrategy.ONE_IN_FOUR,
            execution_configs.FSDPWrapStrategy.ONE_IN_FIVE,
        ]:
            # Map strategy to divisor
            divisor_map = {
                execution_configs.FSDPWrapStrategy.ONE_IN_TWO: 2,
                execution_configs.FSDPWrapStrategy.ONE_IN_THREE: 3,
                execution_configs.FSDPWrapStrategy.ONE_IN_FOUR: 4,
                execution_configs.FSDPWrapStrategy.ONE_IN_FIVE: 5,
            }
            divisor = divisor_map[strategy]

            blocks = self.get_transformer_blocks()
            blocks_to_wrap = set(blocks[i] for i in range(0, len(blocks), divisor))

            def wrap_policy(
                module: nn.Module,
                recurse: bool,
                nonwrapped_numel: int,
            ) -> bool:
                if recurse:
                    return True
                return module in blocks_to_wrap

            return wrap_policy

        else:
            raise ValueError(f"Unknown wrapping strategy: {strategy}")

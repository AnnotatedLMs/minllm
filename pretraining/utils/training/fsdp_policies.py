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

    Architecture classes should inherit from this and implement:
    - get_fsdp_wrappable_modules(): Module types that can be wrapped
    - get_transformer_blocks(): List of transformer blocks
    - get_fsdp_special_modules(): Special modules like embeddings/output layers
    """

    def get_fsdp_wrappable_modules(self) -> typing.Set[typing.Type[nn.Module]]:
        """
        Return module TYPES that FSDP should wrap.

        Used by: BY_BLOCK, BY_BLOCK_AND_SIZE strategies
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

        Used by: ONE_IN_TWO, ONE_IN_THREE, etc. strategies, and block grouping
        Used when: We need to wrap specific blocks (e.g., every 2nd block)
                   or group blocks together for FSDP.

        Returns:
            List of transformer block instances (in order)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_transformer_blocks()"
        )

    def get_fsdp_special_modules(self) -> typing.Dict[str, nn.Module]:
        """
        Return large module INSTANCES that should be wrapped separately.

        Used by: BY_BLOCK_AND_SIZE, BY_BLOCK_GROUP_AND_SIZE strategies
        Used when: We want to wrap embeddings/output layers separately
                   from transformer blocks for memory efficiency.

        Note: Be careful with weight-tied modules (e.g., if lm_head shares weights
              with token_embeddings, only include one to avoid double-wrapping).

        Returns:
            Dict mapping module names to module instances
        """
        # Default implementation - architectures can override if needed
        special_modules = {}

        # Common pattern: token embeddings
        if hasattr(self, "token_embeddings"):
            special_modules["token_embeddings"] = self.token_embeddings

        # Common pattern: output head/lm_head
        if hasattr(self, "lm_head"):
            special_modules["lm_head"] = self.lm_head
        elif hasattr(self, "output_head"):
            special_modules["output_head"] = self.output_head

        return special_modules

    def prepare_for_fsdp(self, config: execution_configs.FSDPConfig) -> None:
        """
        Prepare the model for FSDP wrapping.

        This may involve creating block groups or other restructuring.

        Args:
            config: FSDP configuration
        """
        if config.wrapping_strategy in [
            execution_configs.FSDPWrapStrategy.BY_BLOCK_GROUP,
            execution_configs.FSDPWrapStrategy.BY_BLOCK_GROUP_AND_SIZE,
        ]:
            # Group blocks together
            blocks = self.get_transformer_blocks()
            if config.block_group_size > 1 and len(blocks) > 0:
                # Create block groups
                block_groups = []
                for i in range(0, len(blocks), config.block_group_size):
                    group_blocks = blocks[i : i + config.block_group_size]
                    group = block_group.BlockGroup(
                        group_blocks,
                        layer_offset=i,
                        activation_checkpointing_strategy=config.activation_checkpointing,
                        activation_checkpointing_reentrant=config.activation_checkpointing_reentrant,
                    )
                    block_groups.append(group)

                # Replace the blocks in the model with block groups
                self._replace_blocks_with_groups(block_groups)

    def _replace_blocks_with_groups(
        self, block_groups: typing.List[block_group.BlockGroup]
    ) -> None:
        """
        Replace transformer blocks with block groups.

        This should be overridden by architecture classes to handle their
        specific structure.

        Args:
            block_groups: List of BlockGroup instances
        """
        # Default implementation for common pattern
        if hasattr(self, "blocks") and isinstance(self.blocks, nn.ModuleList):
            # Clear existing blocks
            self.blocks = nn.ModuleList(block_groups)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement _replace_blocks_with_groups() "
                f"or have a 'blocks' attribute of type nn.ModuleList"
            )

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

# Standard Library
import typing
from dataclasses import dataclass
from functools import cached_property

# Third Party
import transformers
from transformers import AutoConfig
from transformers import AutoTokenizer

# Project
from posttraining.instruction_tuning.common.patterns.tokenization import chat_setup
from posttraining.instruction_tuning.common.patterns.tokenization import special_tokens
from posttraining.instruction_tuning.utils import hashing

DEFAULT_SFT_MESSAGES_KEY = "messages"
GROUND_TRUTHS_KEY = "ground_truth"


@dataclass
class TokenizerConfig:
    """Configuration for tokenizers used in instruction tuning."""

    tokenizer_name_or_path: typing.Optional[str] = None
    tokenizer_revision: typing.Optional[str] = None
    trust_remote_code: bool = False
    use_fast: bool = True
    chat_template_name: str = "tulu"
    add_bos: bool = False

    # For tracking purposes
    tokenizer_files_hash: typing.Optional[typing.List[str]] = None

    # Message keys
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY

    @cached_property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Get the configured tokenizer with proper special tokens and chat template."""
        # Track tokenizer files for cache invalidation
        files_hash = hashing.get_files_hash_if_exists(
            self.tokenizer_name_or_path,
            self.tokenizer_revision,
            filenames=[
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "vocab.json",
            ],
        )
        self.tokenizer_files_hash = ",".join(files_hash)

        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            revision=self.tokenizer_revision,
            trust_remote_code=self.trust_remote_code,
            use_fast=self.use_fast,
        )

        # Get model config for validation
        config = AutoConfig.from_pretrained(
            self.tokenizer_name_or_path, revision=self.tokenizer_revision
        )

        # Validate model-specific requirements
        special_tokens.validate_model_specific_requirements(
            tokenizer=tokenizer,
            model_type=config.model_type,
            chat_template_name=self.chat_template_name,
            add_bos=self.add_bos,
            use_fast=self.use_fast,
        )

        # Configure padding token
        tokenizer = special_tokens.configure_padding_token(tokenizer)

        # Configure chat template
        tokenizer = chat_setup.configure_chat_template(
            tokenizer=tokenizer,
            chat_template_name=self.chat_template_name,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer_revision=self.tokenizer_revision,
            add_bos=self.add_bos,
        )

        return tokenizer

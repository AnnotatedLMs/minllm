# Standard Library
import dataclasses


@dataclasses.dataclass
class MultiTokenPredictionConfig:
    """Configuration for multi-token prediction heads."""

    n_predict: int = 3  # Number of future tokens to predict
    prediction_depth: int = 1  # Number of layers in prediction head
    dropout_rate: float = 0.1  # Dropout in prediction head


@dataclasses.dataclass
class OutputHeadConfig:
    """Configuration for output projection heads."""

    # Language modeling head
    tie_word_embeddings: bool = True  # Whether to tie input/output embeddings
    lm_head_bias: bool = False  # Whether to use bias in LM head

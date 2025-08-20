# Project
from pretraining.configs import base


class MultiTokenPredictionConfig(base.BaseConfig):
    """Configuration for multi-token prediction heads."""

    n_predict: int = 3  # Number of future tokens to predict
    prediction_depth: int = 1  # Number of layers in prediction head
    dropout: float = 0.1  # Dropout in prediction head


class OutputHeadConfig(base.BaseConfig):
    """Configuration for output projection heads."""

    # Language modeling head
    tie_word_embeddings: bool = True  # Whether to tie input/output embeddings
    lm_head_bias: bool = False  # Whether to use bias in LM head

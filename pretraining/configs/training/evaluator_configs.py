# Third Party
import pydantic

# Project
from pretraining.configs import base


class EvaluatorConfig(base.BaseConfig):
    """Consumed by: Evaluation loop"""

    eval_interval: int = pydantic.Field(gt=0)
    eval_iters: int = pydantic.Field(gt=0)
    num_eval_batches: int = pydantic.Field(
        default=50, gt=0, description="Number of batches to evaluate on"
    )

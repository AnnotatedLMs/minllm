# Standard Library
import dataclasses


@dataclasses.dataclass
class EvaluatorConfig:
    """Consumed by: Evaluation loop"""

    eval_interval: int
    eval_iters: int
    eval_only: bool

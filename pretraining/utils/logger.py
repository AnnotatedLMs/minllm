# Standard Library
import logging
import pathlib
import sys
import typing

# Third Party
import torch.distributed as dist


def setup_logging(
    log_dir: typing.Optional[pathlib.Path] = None, log_level: str = "INFO", rank: int = 0
) -> logging.Logger:
    """Setup logging configuration for training.

    Args:
        log_dir: Directory to save log files (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rank: Process rank for distributed training

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("minllm")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler - only for rank 0
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler - only for rank 0
    if log_dir is not None and rank == 0:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"minllm.{name}")


def log_rank_0(logger: logging.Logger, message: str, level: str = "info") -> None:
    """Log message only on rank 0 process.

    In distributed training, multiple processes run in parallel.
    This ensures only the main process (rank 0) outputs logs to
    avoid duplicate messages.

    Args:
        logger: Logger instance
        message: Message to log
        level: Log level (debug, info, warning, error)
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        getattr(logger, level)(message)

# Standard Library
import os
import typing

# Third Party
from accelerate import Accelerator
from peft import PeftModel
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer


def get_last_checkpoint_path(output_dir: str, incomplete: bool = False) -> typing.Optional[str]:
    """
    Get the path to the last checkpoint.

    Args:
        output_dir: Output directory containing checkpoints
        incomplete: Whether to include incomplete checkpoints

    Returns:
        Path to last checkpoint or None
    """
    if not os.path.exists(output_dir):
        return None

    # Find all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and (item.startswith("step_") or item.startswith("epoch_")):
            # Check if checkpoint is complete
            if not incomplete and not os.path.exists(os.path.join(item_path, "COMPLETED")):
                continue
            checkpoint_dirs.append(item_path)

    if not checkpoint_dirs:
        return None

    # Sort by modification time and return the latest
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x))
    return checkpoint_dirs[-1]


def save_model_checkpoint(
    accelerator: Accelerator,
    model: typing.Union[PreTrainedModel, PeftModel],
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    use_lora: bool = False,
):
    """
    Save model checkpoint with proper handling for LoRA and distributed training.

    Args:
        accelerator: Accelerator instance
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        use_lora: Whether model uses LoRA
    """
    accelerator.wait_for_everyone()

    # Unwrap model if needed
    unwrapped_model = accelerator.unwrap_model(model)

    # Save on main process
    if accelerator.is_main_process:
        if use_lora:
            # Save LoRA weights
            unwrapped_model.save_pretrained(output_dir)
        else:
            # Save full model
            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=True,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)

    accelerator.wait_for_everyone()


def clean_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
):
    """
    Clean old checkpoints, keeping only the last N.

    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep (-1 keeps all)
    """
    if keep_last_n < 0:
        return

    # Find all checkpoint directories
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and (item.startswith("step_") or item.startswith("epoch_")):
            checkpoint_dirs.append(item_path)

    # Sort by modification time
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x))

    # Remove old checkpoints
    checkpoints_to_remove = checkpoint_dirs[:-keep_last_n] if keep_last_n > 0 else checkpoint_dirs
    for checkpoint_dir in checkpoints_to_remove:
        print(f"Removing old checkpoint: {checkpoint_dir}")
        # Standard Library
        import shutil

        shutil.rmtree(checkpoint_dir)


def parse_checkpoint_step(checkpoint_path: str) -> typing.Tuple[int, int, int, int]:
    """
    Parse checkpoint path to extract training state.

    Args:
        checkpoint_path: Path to checkpoint (e.g., "step_100" or "epoch_5")

    Returns:
        Tuple of (starting_epoch, resume_batch_idx, resume_step, completed_steps)
    """
    checkpoint_name = os.path.basename(checkpoint_path)
    training_difference = os.path.splitext(checkpoint_name)[0]

    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_batch_idx = 0
        completed_steps = 0  # Will be calculated later
        resume_step = 0
    else:
        # Extract step number
        step_num = int(training_difference.replace("step_", ""))
        resume_batch_idx = step_num  # Will be multiplied by gradient_accumulation_steps
        starting_epoch = 0  # Will be calculated later
        completed_steps = step_num
        resume_step = 0  # Will be calculated later

    return starting_epoch, resume_batch_idx, resume_step, completed_steps

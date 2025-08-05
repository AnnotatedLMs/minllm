#!/usr/bin/env python3
# Standard Library
import logging
import os

# Third Party
import torch
import transformers
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import HfArgumentParser

# Project
from posttraining.instruction_tuning.common.patterns.collation import padding_collator
from posttraining.instruction_tuning.common.patterns.trainers import sft_trainer
from posttraining.instruction_tuning.configs.data import tokenizer_config
from posttraining.instruction_tuning.configs.training import sft_arguments
from posttraining.instruction_tuning.data import dataset_builder
from posttraining.instruction_tuning.data.transforms import sft_transforms
from posttraining.instruction_tuning.data.transforms import visualization

logger = get_logger(__name__)


def main():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser((sft_arguments.SFTArguments, tokenizer_config.TokenizerConfig))
    args, tokenizer_config = parser.parse_args_into_dataclasses()

    # Initialize accelerator
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to if args.report_to != "all" else None,
        project_dir=args.output_dir,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load tokenizer
    tokenizer = tokenizer_config.tokenizer

    # Load and prepare dataset
    with accelerator.main_process_first():
        transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
        train_dataset, dataset_stats = dataset_builder.get_cached_sft_dataset(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tokenizer_config,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=sft_transforms.TOKENIZED_SFT_DATASET_KEYS,
            dataset_local_cache_dir="local_dataset_cache",
        )
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")

    # Visualize first example
    if accelerator.is_main_process:
        logger.info("First training example:")
        visualization.visualize_token(train_dataset[0]["input_ids"], tokenizer)

    # Load model configuration
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=tokenizer_config.trust_remote_code,
    )

    # Load model
    if args.use_qlora:
        # QLoRA configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_index = accelerator.local_process_index
        device_map = {"": device_index}

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            config=config,
            trust_remote_code=tokenizer_config.trust_remote_code,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            config=config,
            trust_remote_code=tokenizer_config.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        )

    # Resize embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Apply LoRA if requested
    if args.use_lora:
        logger.info("Initializing LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Create data collator
    collate_fn = padding_collator.create_sft_collator(
        tokenizer=tokenizer,
        packing=args.packing,
        model=model,
    )

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )

    # Create optimizer using factory
    # Project
    from posttraining.instruction_tuning.utils.training import optimizer_factory

    optimizer = optimizer_factory.OptimizerFactory.create_adamw(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_optimizer,
        use_qlora=args.use_qlora,
    )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Create learning rate scheduler using factory
    # Project
    from posttraining.instruction_tuning.utils.training import scheduler_factory

    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = scheduler_factory.SchedulerFactory.create_from_name(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Create trainer
    trainer = sft_trainer.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
    )

    # Train
    trainer.train()

    # End training
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()

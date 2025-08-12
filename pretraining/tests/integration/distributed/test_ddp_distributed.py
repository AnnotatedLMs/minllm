# Standard Library
import unittest

# Third Party
import torch
import torch.distributed as dist

# Project
from pretraining.common.patterns.architectures import deepseek3
from pretraining.common.patterns.architectures import gpt2
from pretraining.common.patterns.architectures import llama3
from pretraining.configs import loader
from pretraining.configs.model.architectures import deepseek
from pretraining.configs.model.architectures import gpt
from pretraining.configs.model.architectures import llama
from pretraining.trainer import llm_trainer
from pretraining.utils.training import loss
from pretraining.utils.training import lr_scheduler
from pretraining.utils.training import optimizer


def create_dummy_dataset(num_samples: int = 100, seq_length: int = 128):
    """Create a dummy dataset for testing."""
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return {"input_ids": input_ids[idx], "labels": input_ids[idx]}

    return DummyDataset()


class TestDDPTraining(unittest.TestCase):
    """Test DDP training with real distributed environment."""

    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment."""
        # Simply check if distributed is already initialized by torchrun
        if not dist.is_initialized():
            raise unittest.SkipTest(
                "Distributed not initialized. Run with:\n"
                "torchrun --nproc_per_node=2 test_ddp_distributed.py"
            )

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()
        cls.device = torch.device(f"cuda:{cls.rank}")
        torch.cuda.set_device(cls.device)

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_gpt2_ddp_training(self):
        """Test GPT-2 DDP training."""
        # Load config
        config_path = "pretraining/configs/examples/debug/gpt2/gpt2_debug_ddp.yaml"
        trainer_config = loader.load_training_config(config_path, gpt.GPT2Config)

        # Create model
        model = gpt2.GPT2.from_config(trainer_config.llm)
        model = model.to(self.device)

        # Wrap with DDP
        dist_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
        )

        # Create optimizer
        optim = optimizer.OptimizerFactory.create_from_config(
            dist_model, trainer_config.training.optimizer, device_type=self.device.type
        )

        # Create scheduler
        scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
            optim,
            trainer_config.training.lr_schedule,
            num_training_steps=trainer_config.training.max_iters,
        )

        # Create data with DistributedSampler
        dataset = create_dummy_dataset(100, 128)
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=trainer_config.training.batch.batch_size,
            sampler=sampler,
        )

        # Create loss handler
        loss_handler = loss.LossHandler(trainer_config.training.loss)

        # Create trainer
        trainer = llm_trainer.LLMTrainer(
            model=model,
            dist_model=dist_model,
            config=trainer_config.training,
            optimizer=optim,
            scheduler=scheduler,
            loss_handler=loss_handler,
            evaluator=None,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Setup trainer
        trainer.setup()

        # Run a few training steps
        step_count = 0
        for batch in train_dataloader:
            metrics = trainer.train_step(batch)
            step_count += 1
            if step_count >= 3:  # Just run 3 steps for testing
                break

            # Verify gradients are synchronized
            if step_count == 1:
                # Check that loss is finite
                self.assertTrue(
                    torch.isfinite(torch.tensor(metrics.get("train/total_loss", float("inf"))))
                )

        # Cleanup
        trainer.cleanup()

        # Synchronize before finishing
        dist.barrier()

    def test_llama3_ddp_training(self):
        """Test Llama3 DDP training."""
        # Load config
        config_path = "pretraining/configs/examples/debug/llama3/llama3_debug_ddp.yaml"
        trainer_config = loader.load_training_config(config_path, llama.Llama3Config)

        # Create model
        model = llama3.Llama3.from_config(trainer_config.llm)
        model = model.to(self.device)

        # Wrap with DDP
        dist_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
        )

        # Create optimizer
        optim = optimizer.OptimizerFactory.create_from_config(
            dist_model, trainer_config.training.optimizer, device_type=self.device.type
        )

        # Create scheduler
        scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
            optim,
            trainer_config.training.lr_schedule,
            num_training_steps=trainer_config.training.max_iters,
        )

        # Create data with DistributedSampler
        dataset = create_dummy_dataset(100, 128)
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=trainer_config.training.batch.batch_size,
            sampler=sampler,
        )

        # Create loss handler
        loss_handler = loss.LossHandler(trainer_config.training.loss)

        # Create trainer
        trainer = llm_trainer.LLMTrainer(
            model=model,
            dist_model=dist_model,
            config=trainer_config.training,
            optimizer=optim,
            scheduler=scheduler,
            loss_handler=loss_handler,
            evaluator=None,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Setup trainer
        trainer.setup()

        # Run a few training steps
        step_count = 0
        for batch in train_dataloader:
            metrics = trainer.train_step(batch)
            step_count += 1
            if step_count >= 3:
                break

            # Check that loss is finite
            self.assertTrue(
                torch.isfinite(torch.tensor(metrics.get("train/total_loss", float("inf"))))
            )

        # Cleanup
        trainer.cleanup()

        # Synchronize
        dist.barrier()

    def test_deepseek3_ddp_training(self):
        """Test DeepSeek3 DDP training with MoE."""
        # Load config
        config_path = "pretraining/configs/examples/debug/deepseek3/deepseek3_debug_ddp.yaml"
        trainer_config = loader.load_training_config(config_path, deepseek.DeepSeek3Config)

        # Create model
        model = deepseek3.DeepSeek3.from_config(trainer_config.llm)
        model = model.to(self.device)

        # Wrap with DDP - MoE models need find_unused_parameters=True
        dist_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=True,  # Required for MoE
        )

        # Create optimizer
        optim = optimizer.OptimizerFactory.create_from_config(
            dist_model, trainer_config.training.optimizer, device_type=self.device.type
        )

        # Create scheduler
        scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
            optim,
            trainer_config.training.lr_schedule,
            num_training_steps=trainer_config.training.max_iters,
        )

        # Create data with DistributedSampler
        dataset = create_dummy_dataset(100, 128)
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=trainer_config.training.batch.batch_size,
            sampler=sampler,
        )

        # Create loss handler
        loss_handler = loss.LossHandler(trainer_config.training.loss)

        # Create trainer
        trainer = llm_trainer.LLMTrainer(
            model=model,
            dist_model=dist_model,
            config=trainer_config.training,
            optimizer=optim,
            scheduler=scheduler,
            loss_handler=loss_handler,
            evaluator=None,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Setup trainer
        trainer.setup()

        # Run a few training steps
        step_count = 0
        for batch in train_dataloader:
            metrics = trainer.train_step(batch)
            step_count += 1
            if step_count >= 3:
                break

            # Check that loss is finite
            self.assertTrue(
                torch.isfinite(torch.tensor(metrics.get("train/total_loss", float("inf"))))
            )

            # Check MoE auxiliary loss if present
            if "train/moe_aux_loss" in metrics:
                self.assertTrue(torch.isfinite(torch.tensor(metrics["train/moe_aux_loss"])))

        # Cleanup
        trainer.cleanup()

        # Synchronize
        dist.barrier()


if __name__ == "__main__":
    unittest.main()

# Standard Library
import unittest

# Third Party
import torch

# Project
from pretraining.common.models.architectures import gpt2
from pretraining.configs import loader
from pretraining.configs.model.architectures import gpt
from pretraining.trainer import llm_trainer
from pretraining.utils.training import dist_utils
from pretraining.utils.training import evaluation
from pretraining.utils.training import loss
from pretraining.utils.training import lr_scheduler
from pretraining.utils.training import optimizer


def create_dummy_dataset(num_samples: int = 100, seq_length: int = 128):
    """Create a dummy dataset for testing."""
    input_ids = torch.randint(0, 1000, (num_samples, seq_length))

    # Create a simple dataset that returns dicts
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return {
                "input_ids": input_ids[idx],
                "labels": input_ids[idx],  # Use same as input_ids for simplicity
            }

    return DummyDataset()


class TestCPUTraining(unittest.TestCase):
    """Test single CPU device training."""

    def setUp(self):
        """Set up test configuration."""
        # Load GPT-2 CPU debug config
        config_path = "pretraining/configs/examples/debug/gpt2_debug_cpu.yaml"
        self.trainer_config = loader.load_training_config(config_path, gpt.GPT2Config)
        self.model_config = self.trainer_config.llm
        self.training_config = self.trainer_config.training

    def test_cpu_training_loop(self):
        """Test basic CPU training loop."""
        device = torch.device("cpu")

        # Create model
        model = gpt2.GPT2.from_config(self.model_config)
        model = model.to(device)

        # Wrap model
        dist_model = dist_utils.SingleAccelerator(model)

        # Create optimizer
        optim = optimizer.OptimizerFactory.create_from_config(
            dist_model, self.training_config.optimizer, device_type=device.type
        )

        # Create scheduler
        scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
            optim,
            self.training_config.lr_schedule,
            num_training_steps=self.training_config.max_iters,
        )

        # Create dummy data
        train_data = create_dummy_dataset(20, 128)
        val_data = create_dummy_dataset(10, 128)

        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.training_config.batch.batch_size,
            shuffle=True,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.training_config.batch.batch_size,
            shuffle=False,
        )

        # Create loss handler and evaluator
        loss_handler = loss.LossHandler(self.training_config.loss)
        evaluator = evaluation.Evaluator(
            loss_handler=loss_handler,
            num_eval_batches=self.training_config.evaluation.num_eval_batches,
        )

        # Create trainer
        trainer = llm_trainer.LLMTrainer(
            model=model,
            dist_model=dist_model,
            config=self.trainer_config,
            optimizer=optim,
            scheduler=scheduler,
            loss_handler=loss_handler,
            evaluator=evaluator,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            rank=0,
            world_size=1,
        )

        # Setup trainer
        trainer.setup()

        # Run training steps
        initial_loss = None
        for batch in train_dataloader:
            metrics = trainer.train_step(batch)
            if initial_loss is None:
                initial_loss = metrics.get("loss", float("inf"))

        # Check that training ran
        self.assertIsNotNone(initial_loss)

        # Cleanup
        trainer.cleanup()


class TestSingleGPUTraining(unittest.TestCase):
    """Test single GPU training."""

    def setUp(self):
        """Set up test configuration."""
        # Load GPT-2 single GPU debug config
        config_path = "pretraining/configs/examples/debug/gpt2/gpt2_debug_single_gpu.yaml"
        self.trainer_config = loader.load_training_config(config_path, gpt.GPT2Config)
        self.model_config = self.trainer_config.llm
        self.training_config = self.trainer_config.training

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_single_gpu_training(self):
        """Test single GPU training."""
        device = torch.device("cuda:0")

        # Create model
        model = gpt2.GPT2.from_config(self.model_config)
        model = model.to(device)

        # Wrap model
        dist_model = dist_utils.SingleAccelerator(model)

        # Create optimizer
        optim = optimizer.OptimizerFactory.create_from_config(
            dist_model, self.training_config.optimizer, device_type=device.type
        )

        # Create scheduler
        scheduler = lr_scheduler.LRSchedulerFactory.create_from_config(
            optim,
            self.training_config.lr_schedule,
            num_training_steps=self.training_config.max_iters,
        )

        # Create dummy data
        train_data = create_dummy_dataset(20, 128)

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.training_config.batch.batch_size,
            shuffle=True,
        )

        # Create loss handler
        loss_handler = loss.LossHandler(self.training_config.loss)

        # Create trainer
        trainer = llm_trainer.LLMTrainer(
            model=model,
            dist_model=dist_model,
            config=self.trainer_config,
            optimizer=optim,
            scheduler=scheduler,
            loss_handler=loss_handler,
            evaluator=None,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            device=device,
            rank=0,
            world_size=1,
        )

        # Setup trainer
        trainer.setup()

        # Run training steps
        for batch in train_dataloader:
            _ = trainer.train_step(batch)

        # Cleanup
        trainer.cleanup()


if __name__ == "__main__":
    unittest.main()

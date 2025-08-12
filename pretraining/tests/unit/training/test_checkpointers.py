# Standard Library
import pathlib
import tempfile
import unittest

# Third Party
import torch
import torch.nn as nn

# Project
from pretraining.configs import core
from pretraining.configs.model import transformer
from pretraining.configs.model.architectures import base as base_llm
from pretraining.configs.model.components import attention
from pretraining.configs.model.components import embeddings
from pretraining.configs.model.components import feedforward
from pretraining.configs.model.components import heads
from pretraining.configs.model.components import normalization
from pretraining.configs.training import batch_configs
from pretraining.configs.training import checkpointer_configs
from pretraining.configs.training import data_configs
from pretraining.configs.training import evaluator_configs
from pretraining.configs.training import execution_configs
from pretraining.configs.training import loss_configs
from pretraining.configs.training import lr_configs
from pretraining.configs.training import optimizer_configs
from pretraining.configs.training import precision_configs
from pretraining.configs.training import system_configs
from pretraining.configs.training import trainer_configs
from pretraining.configs.training import wandb_configs
from pretraining.trainer import checkpoint_data
from pretraining.utils.training.checkpointers import base_checkpointer
from pretraining.utils.training.checkpointers import core_checkpointer


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        return self.linear2(x)


class TestCheckpointers(unittest.TestCase):
    """Test checkpointer implementations."""

    def setUp(self):
        """Set up test configuration."""
        self.config = checkpointer_configs.CheckpointerConfig(
            save_dir="/tmp/test_checkpoints",
            save_interval=100,
            keep_last_n=2,
            save_best=False,
        )
        self.model = DummyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _create_dummy_llm_config(self):
        """Create a minimal LLM config for testing."""
        return base_llm.BaseLLMConfig(
            vocab_size=1000,
            token_embedding=embeddings.TokenEmbeddingConfig(
                embedding_dim=128,
                init_std=0.02,
            ),
            transformer=transformer.TransformerConfig(
                hidden_dim=128,
                n_layers=2,
                block_size=128,  # Match sequence_length in batch config
                normalization=normalization.LayerNormConfig(
                    norm_eps=1e-5,
                    bias=True,
                ),
                attention=attention.MultiHeadAttentionConfig(
                    num_heads=4,
                    bias=True,
                    max_seq_length=128,
                    is_causal=True,
                    use_flash_attention=False,
                ),
                ffn=feedforward.FFNConfig(
                    activation="gelu",
                    bias=True,
                    intermediate_dim=512,
                ),
                dropout=0.0,
                bias=True,
            ),
            output_head=heads.OutputHeadConfig(
                tie_word_embeddings=True,
                lm_head_bias=False,
            ),
        )

    def _create_test_config(self):
        """Create a minimal TrainingLoopConfig for testing."""
        return trainer_configs.TrainingLoopConfig(
            max_iters=1000,
            data=data_configs.DataConfig(
                dataset="test_dataset",
                data_dir="/tmp/test_data",
                num_workers=0,
            ),
            batch=batch_configs.BatchConfig(batch_size=8, sequence_length=128),
            optimizer=optimizer_configs.OptimizerConfig(
                optimizer_type="adamw",
                learning_rate=0.001,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.999,
                grad_clip=1.0,
                eps=1e-8,
                parameter_grouping="dimension",
            ),
            lr_schedule=lr_configs.LearningRateScheduleConfig(
                schedule_type="cosine_with_warmup",
                warmup_iters=100,
                lr_decay_iters=1000,
                min_lr=1e-5,
            ),
            loss=loss_configs.LossConfig(),
            evaluation=evaluator_configs.EvaluatorConfig(
                eval_interval=100,
                eval_iters=10,
            ),
            checkpoint=self.config,
            torch_compilation=system_configs.TorchCompilationConfig(compile=False),
            execution=execution_configs.ExecutionConfig(),
            precision=precision_configs.PrecisionType.FP32,
            log_interval=10,
            wandb_logging=wandb_configs.WandbConfig(
                enabled=False,
            ),
            seed=42,
        )

    def test_base_checkpointer_abstract(self):
        """Test that BaseCheckpointer is abstract."""
        with self.assertRaises(TypeError):
            base_checkpointer.BaseCheckpointer(self.config)

    def test_core_checkpointer_save_load(self):
        """Test CoreCheckpointer save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Update config with temp directory
            self.config.save_dir = tmpdir
            checkpointer = core_checkpointer.Checkpointer(self.config)

            # Create minimal config for testing
            test_config = self._create_test_config()

            # Create checkpoint data
            ckpt_data = checkpoint_data.CheckpointData(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state={"last_epoch": 0},
                training_state={"iteration": 100, "best_val_loss": 1.5},
            )

            # Create full trainer config for save_checkpoint
            trainer_config = core.TrainerConfig(
                llm=self._create_dummy_llm_config(),
                training=test_config,
            )

            # Save checkpoint
            checkpointer.save_checkpoint(ckpt_data, trainer_config)

            # Check that checkpoint was saved as directory
            checkpoint_path = pathlib.Path(tmpdir) / "step100"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(checkpoint_path.is_dir())

            # Check that all component files exist
            self.assertTrue((checkpoint_path / "config.yaml").exists())
            self.assertTrue((checkpoint_path / "model.pt").exists())
            self.assertTrue((checkpoint_path / "optim.pt").exists())
            self.assertTrue((checkpoint_path / "scheduler.pt").exists())
            self.assertTrue((checkpoint_path / "train.pt").exists())

            # Load checkpoint
            loaded_ckpt = checkpointer.load_checkpoint("cpu")
            self.assertIsNotNone(loaded_ckpt)
            self.assertEqual(loaded_ckpt.training_state["iteration"], 100)

    def test_core_checkpointer_rotation(self):
        """Test checkpoint rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.save_dir = tmpdir
            self.config.keep_last_n = 2
            checkpointer = core_checkpointer.Checkpointer(self.config)

            # Create test config
            test_config = self._create_test_config()

            # Save 3 checkpoints
            for i in [100, 200, 300]:
                ckpt_data = checkpoint_data.CheckpointData(
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state={"last_epoch": i // 100},
                    training_state={"iteration": i, "best_val_loss": 1.5},
                )
                trainer_config = core.TrainerConfig(
                    llm=self._create_dummy_llm_config(),
                    training=test_config,
                )
                checkpointer.save_checkpoint(ckpt_data, trainer_config)

            # Check that only last 2 checkpoints exist
            checkpoint_dir = pathlib.Path(tmpdir)
            checkpoints = list(checkpoint_dir.glob("step*"))
            self.assertEqual(len(checkpoints), 2)

            # Check that the oldest checkpoint was removed
            self.assertFalse((checkpoint_dir / "step100").exists())
            self.assertTrue((checkpoint_dir / "step200").exists())
            self.assertTrue((checkpoint_dir / "step300").exists())

    def test_checkpoint_data_creation(self):
        """Test CheckpointData creation and field access."""
        ckpt_data = checkpoint_data.CheckpointData(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state={"last_epoch": 10},
            training_state={"iteration": 100, "best_val_loss": 1.5},
        )

        # Check that all fields are accessible
        self.assertIsNotNone(ckpt_data.model_state)
        self.assertIsNotNone(ckpt_data.optimizer_state)
        self.assertIsNotNone(ckpt_data.scheduler_state)
        self.assertIsNotNone(ckpt_data.training_state)

        # Check specific values
        self.assertEqual(ckpt_data.scheduler_state["last_epoch"], 10)
        self.assertEqual(ckpt_data.training_state["iteration"], 100)
        self.assertEqual(ckpt_data.training_state["best_val_loss"], 1.5)

        # Check that model state dict has the expected keys
        model_keys = set(ckpt_data.model_state.keys())
        expected_keys = {
            "embedding.weight",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
        }
        self.assertEqual(model_keys, expected_keys)


class TestCheckpointerUtils(unittest.TestCase):
    """Test checkpointer utility functions."""

    def test_best_checkpoint_tracking(self):
        """Test tracking of best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = checkpointer_configs.CheckpointerConfig(
                save_dir=tmpdir,
                save_interval=100,
                keep_last_n=2,
                save_best=True,
            )
            checkpointer = core_checkpointer.Checkpointer(config)

            # First checkpoint - should be best
            self.assertTrue(checkpointer.should_save_best(1.5))

            # Worse checkpoint - should not be best
            self.assertFalse(checkpointer.should_save_best(2.0))

            # Better checkpoint - should be best
            self.assertTrue(checkpointer.should_save_best(1.0))

    def test_find_resume_checkpoint(self):
        """Test finding checkpoint to resume from."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = checkpointer_configs.CheckpointerConfig(
                save_dir=tmpdir,
                save_interval=100,
                keep_last_n=2,
                save_best=False,
            )
            checkpointer = core_checkpointer.Checkpointer(config)

            # No checkpoint exists
            resume_path = checkpointer.find_resume_checkpoint()
            self.assertIsNone(resume_path)

            # Create a checkpoint directory
            checkpoint_path = pathlib.Path(tmpdir) / "step100"
            checkpoint_path.mkdir(parents=True)
            torch.save({"test": "data"}, checkpoint_path / "model.pt")

            # Create latest symlink
            latest_link = pathlib.Path(tmpdir) / "latest"
            latest_link.symlink_to(checkpoint_path.name)

            # Should find the checkpoint
            resume_path = checkpointer.find_resume_checkpoint()
            self.assertIsNotNone(resume_path)
            self.assertEqual(resume_path.name, "step100")

    def test_explicit_resume_path(self):
        """Test resuming from explicit path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an actual checkpoint file
            checkpoint_path = pathlib.Path(tmpdir) / "resume_checkpoint.pt"
            torch.save({"test": "data"}, checkpoint_path)

            config = checkpointer_configs.CheckpointerConfig(
                save_dir=tmpdir,
                save_interval=100,
                keep_last_n=2,
                save_best=False,
                resume_from=str(checkpoint_path),
            )
            checkpointer = core_checkpointer.Checkpointer(config)

            # Should use explicit path
            resume_path = checkpointer.find_resume_checkpoint()
            self.assertIsNotNone(resume_path)
            self.assertEqual(str(resume_path), str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()

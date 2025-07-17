# Configuration Discrepancies Analysis

## Summary of Issues Found

After analyzing the config files, parsers, and YAML examples, here are the discrepancies that need to be fixed:

### 1. **Data Configuration Issues**

**Config Class (`data_configs.py`):**
- Has fields: `dataset`, `data_dir` (with default='data')
- Missing fields that parser expects

**Parser Expects (`training_parser.py` lines 64-69):**
- `dataset`
- `data_dir`
- `num_workers` ❌ (NOT in config class)
- `pin_memory` ❌ (NOT in config class)

**YAML Provides:**
- `dataset_name` (not `dataset`) ❌
- `data_dir`
- `train_split` ❌ (NOT expected by parser)
- `val_split` ❌ (NOT expected by parser)
- `num_workers` ✓ (expected by parser but not in config)

**Fix Required:**
- Add `num_workers` and `pin_memory` to `DataConfig` class
- Parser should handle `dataset_name` → `dataset` mapping
- Add `train_split` and `val_split` to config if needed

### 2. **System Configuration Issues**

**TorchCompilationConfig Issues:**
- Config class has: `compile` (default=True)
- Parser expects: `compile`, `compile_mode` ❌ (NOT in config class)
- YAML provides: only `compile`

**DistributedConfig Issues:**
- Config class has: `backend` (default='nccl')
- Parser expects: `backend`, `ddp_bucket_cap_mb` ❌, `find_unused_parameters` ❌
- YAML provides: only `distributed` (boolean)

**DeviceConfig Issues:**
- Config class has: `device`, `dtype`
- Parser expects: `device`, `dtype`
- YAML provides: `device`, `torch_dtype` (not `dtype`) ❌

**Fix Required:**
- Add missing fields to config classes
- Handle YAML naming differences (torch_dtype → dtype)
- Handle boolean `distributed` → full config conversion

### 3. **Checkpointer Configuration Issues**

**Config Class (`checkpointer_configs.py`):**
- Has: `always_save_checkpoint`, `init_from`
- Missing many fields parser expects

**Parser Expects (`training_parser.py` lines 103-111):**
- `save_dir` ❌ (NOT in config)
- `save_interval` ❌ (NOT in config)
- `save_best` ❌ (NOT in config)
- `keep_last_n` ❌ (NOT in config)
- `resume_from` ❌ (NOT in config)

**YAML Provides:**
- `save_dir`
- `save_interval`
- `keep_last_n`
- Not providing: `save_best`, `resume_from`

**Fix Required:**
- Add all missing fields to `CheckpointerConfig`
- Provide defaults for optional fields

### 4. **Evaluator Configuration Issues**

**Config Class (`evaluator_configs.py`):**
- Has: `eval_interval`, `eval_iters`, `eval_only`
- No defaults provided (all required)

**Parser Expects:**
- All three fields, with `eval_only` having a default of False

**YAML Provides:**
- `eval_interval`
- `eval_iters`
- Not providing: `eval_only`

**Fix Required:**
- Add default for `eval_only` in config class

### 5. **Wandb Configuration Issues**

**Config Class (`wandb_configs.py`):**
- Has: `out_dir`, `wandb_log`, `wandb_project`, `wandb_run_name`

**Parser Expects (`training_parser.py` lines 93-101):**
- `enabled` ❌ (config has `wandb_log`)
- `project` ❌ (config has `wandb_project`)
- `run_name` (optional)
- `tags` (optional) ❌ (NOT in config)
- `log_model` ❌ (NOT in config)

**YAML Provides:**
- `use_wandb` (not `enabled` or `wandb_log`) ❌
- `wandb_project`
- `wandb_run_name`

**Fix Required:**
- Standardize field names between config, parser, and YAML
- Add missing fields (`tags`, `log_model`)

### 6. **MoE Training Configuration Issues**

**YAML (deepseek3.yaml) Provides:**
- `load_balancing_loss_weight` (not `aux_loss_weight`) ❌
- `router_temperature` ❌ (NOT in config)
- `router_noise_scale` ❌ (NOT in config)

**Config Class Expects:**
- `aux_loss_weight`
- `capacity_factor`
- `drop_tokens`
- `z_loss_weight`

**Fix Required:**
- Align field names or add mapping logic
- Add missing fields to config

### 7. **MTP Training Configuration Issues**

**YAML (deepseek3.yaml) Provides:**
- `loss_weight` (not `mtp_loss_weight`) ❌
- `warmup_steps` ❌ (NOT in config)

**Config Class Expects:**
- `mtp_loss_weight`

**Fix Required:**
- Align field names
- Add `warmup_steps` if needed

### 8. **Learning Rate Schedule Issues**

**Config Class:**
- Optional fields: `num_cycles`, `phase_steps`, `phase_names`, `decay_lr`
- Required fields: `schedule_type`, `warmup_iters`, `lr_decay_iters`, `min_lr`

**YAML Provides:**
- All required fields ✓
- Not providing optional fields (which is fine)

**No issues here.**

### 9. **Training Section Structure Issues**

**Registry (`registry.py`) Expects in 'training' dict:**
- `batch_size` ✓
- `gradient_accumulation_steps` ✓
- `max_iters` ✓
- `log_interval` ✓
- `eval_interval` ✓
- `eval_iters` ✓
- `seed` ❌ (actually provided at root level)

**YAML Structure:**
- Has 'training' section with most fields
- Has 'seed' at root level, not in training section

**Fix Required:**
- Registry should look for `seed` in root config dict (line 254)

## Priority Fixes

1. **Critical (Blocking Errors):**
   - DataConfig: Add `num_workers`, `pin_memory`
   - CheckpointerConfig: Add all missing fields
   - WandbConfig: Reconcile field name differences
   - System configs: Add missing fields
   - Fix seed location lookup in registry

2. **Important (Feature Gaps):**
   - MoE/MTP training config field alignment
   - Handle YAML naming variations (dataset_name, torch_dtype, etc.)

3. **Nice to Have:**
   - Add validation for optional fields
   - Better error messages for missing fields

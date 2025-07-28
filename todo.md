# TODO

## README
- expectations
- setup
- citations

## PRETRAINING
- TRANSFORMER CONFIG: norm and attention set as base for conevneince but it might be misleading
- TRAINING LOOP CONFIGS: config.training.max_grad_norm & self.config.training.mixed_precision not available

- trainer scripts for each architecture

- use temp2 to see if dataset and dataloader are set up to encapsulate the training needs of each architecture...or if we need slightly diff ones for each architecture
- use olmo to review configs for different nn modules, are we exposing the right stuff

- Add precision config (fp32/bf16/fp16) to training configs
- Add save/load path configs for sophisticated checkpoint loading
- Add gradient clipping warmup (warmup_steps/factor)
- Add max_duration config with token support (e.g. '2e12T')
- Add speed monitoring/MFU tracking config

- position.py Explain buffer vs parameter distinction
- MLA flash attn?

## POSTTRAINING
### Instruction Tuning
- find implementations of instruction sft tuning
- find implementations of the dataset/dataloader

### Preference Tuning
- find implementations of preference tuning
- find implementations of the dataset/dataloader

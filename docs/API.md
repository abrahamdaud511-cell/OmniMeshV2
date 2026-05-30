# API Reference

## OmniMeshV2

### `OmniMeshV2(config)`
Main model class.

#### Methods

- `generate(prompt=None, file_path=None, max_new_tokens=512, temperature=0.7, top_k=50, use_safety=True) -> str`
- `forward(inputs, mode='train') -> torch.Tensor`
- `train_with_file(file_path, epochs=5, learning_rate=3e-4)`

## UniversalDataIngestionEngine

- `ingest_file(file_path) -> torch.Tensor`
- `ingest_directory(dir_path, pattern) -> List[Tuple[str, torch.Tensor]]`

## AdaptiveTrainingStabilityGovernor

- `start()`
- `stop()`
- `get_training_params() -> Dict`

## ConstitutionalSafetyRouterV2

- `check_and_revise(raw_output, context) -> str`

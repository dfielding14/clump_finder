# Configuration Layout

- `base/`: canonical templates such as `configs/base/config.yaml` for quick local runs.
- `presets/`: resolution-specific presets (`config_nXXXX.yaml`) used by SLURM jobs and sweeps.
- `runs/`: grouped campaign configs that feed the batch sbatch wrappers.

All configs are YAML and compatible with `clump_finder.py --config <path>`.

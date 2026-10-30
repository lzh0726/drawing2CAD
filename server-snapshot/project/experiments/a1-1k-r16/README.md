# A1: 1,000 samples, language LoRA rank 16

Training configuration: train-a1-1k.yaml.
Training weights and checkpoints: training-output (link to the original output directory).
The original training launch record is train-launch.json.

Evaluation records:
- baseline-pilot-v1: original base-model control.
- finetuned-pilot-v1-repairs0: initial SFT evaluation, no repairs.
- finetuned-pilot-v1-repairs2: historical repair-enabled evaluation.
- entity-rate-current-20261027-141244: later generation-budget evaluation; inspect protocol before comparison.
- success-geometry-audit-20261027-190624: dimension and geometry analysis.

These are historical records of A1 under different inference protocols; do not combine them as one score.
Record contents are preserved. Embedded historical absolute paths retain their original provenance;
use the cleanup manifest to locate relocated files.

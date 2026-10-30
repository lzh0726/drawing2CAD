# A3: language LoRA + full multimodal projectors

Same base Qwen3-VL-8B, 10,000 training / 100 validation samples, seed 42,
2 epochs (1,250 optimizer steps), rank 16, 4096 tokens, 262144 image pixels,
microbatch 1 / gradient accumulation 16, BF16 and checkpointing as A2.
Starts from the original base model, not A2 weights.
Vision encoder frozen. All four projectors saved through PEFT modules_to_save.
Language LoRA LR 1e-4; full projector LR 1e-5, same cosine scheduler.
This compares an unfreezing recipe including its smaller projector LR.

Watcher checks every 300 seconds. Admission requires three observations 10 seconds apart:
utilization <10%, free memory >42 GiB,
used-memory variation <=512 MiB. Admission revised by user to >42 GiB available memory.
One GPU by UUID, project training.lock, no duplicate launch. No other process is stopped.
PyTorch allocator cap 36 GiB; supervisor checks every 10 seconds, stops only this run if
GPU free memory <4 GiB, data-disk free <10 GiB, or GPU monitoring fails.
This is cooperative monitoring, not a hardware reservation or a throughput guarantee.

CPU meta audit validates exact trainable and serialized parameter counts before enabling watcher.
First optimizer step checks nonzero finite gradients in all four projectors.
Steps 1/5/25, then every 100 steps and validation log peak allocated/reserved memory.
Every checkpoint verifies all projector tensors are present. Failure/OOM stops the run,
does not silently change settings or automatically restart. Longest-sequence peak memory
and successful optimization remain unverified until real training occurs.

No test samples used in training, no original evaluation files or production service changed.
Original data license/review status remains unchanged; not claimed cleared or human reviewed.

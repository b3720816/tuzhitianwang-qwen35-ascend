---
name: qwen35-ascend-migrate
description: Orchestrate Qwen3.5-0.8B Ascend migration preflight and training validation using the project's verified MindSpeed-MM source bundle. Use for migration execution and evidence collection, not business risk scoring.
---

# Qwen3.5 Ascend Migration

Use `scripts/migrate.py` with explicit absolute paths to the verified source bundle,
Python environment, HF model, converted DCP checkpoint, official annotations, CANN
environment script, and a new output directory. This skill depends on the source
bundle; it does not install packages or download models.

1. Inspect the supplied source and environment. Use the 20260927 HF-initialized
   source with the DDP gradient synchronization fix; older snapshots are insufficient.
2. Run the helper without `--execute` to inspect its JSON plan. Missing paths fail
   closed. A plan is not NPU validation.
3. Run `--stage preflight --execute` only on the user's authorized Ascend host.
   Read the generated preflight report and console log. Stop on failure; do not
   install or downgrade dependencies automatically.
4. Use `--stage train --execute --confirm-training` only when the user requests a
   new 100-step run. Existing successful evidence should not trigger retraining.
   Supply `--verified-config` with the successful HF run's `train100.yaml`.
   The new helper preserves its training settings, rewrites model/data paths, and
   isolates cache and checkpoints in a new directory. It rejects legacy DCP/meta
   initialization. It does not compare against the old reference log.
5. Report the receipt and artifacts. Nonzero status is failure; interrupted work is
   not success. Training comparison is not inference or checkpoint restore proof.

Required helper arguments: `--bundle`, `--python`, `--hf-model`,
`--dataset`, `--cann-env`, `--output`. Optional `--stage preflight|train`.
The output path must not already exist when executing. Paths are passed as arguments
and environment values, not interpolated shell commands.
The legacy asset preflight requires `--dcp`; HF training does not. Training requires
PyYAML in the selected environment. The DDP source marker check is a guard, not a
complete source audit. Use only the trusted, verified source bundle.

## Text Inference and Weight Restore

Use `scripts/verify_inference.py --model HF_DIR --output NEW_DIR` to prepare a
read-only plan. On an authorized NPU host, source the CANN environment and append
`--execute`. For your own trusted training checkpoint append `--checkpoint ITER_DIR
--mindspeed-root SOURCE/third_party/MindSpeed-MM --trust-local-checkpoint`.
ITER_DIR must contain `.metadata` and `.distcp` shards. DCP metadata is pickle-based;
do not load untrusted checkpoints. Results record base-model and restored modes
separately, hash source weights, reject CPU fallback, and require strict DCP parameter
loading, finite forward logits, and nonempty generated text. A passed text smoke test
does not verify multimodal quality, optimizer restoration, or resumed training.

Scope boundary: a working CANN/Python runtime and local HF weights are prerequisites.
Automated environment provisioning and HF-to-DCP conversion are not implemented.
The 20260927 cloud evidence verifies 100-step training and text/image smoke tests.
This revised training wrapper has only local orchestration/configuration tests;
do not describe it as newly NPU-tested. Smoke tests do not establish held-out
accuracy or optimizer resume support. Existing evidence does not require retraining.

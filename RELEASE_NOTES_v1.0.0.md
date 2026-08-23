# VEGA-SR v1.0.0 release notes

This paper-aligned release adds the reproducible PSE real-world comparison and
its machine-readable outputs.

## Included

- pinned PSE source commit and raw-data SHA-256 digests;
- EMPS and Roughpipe split definitions;
- 20-seed launchers for PSE and PSE-aligned VEGA-SR;
- PySR and Operon baseline launchers plus the EMPS Physics-LS ablation;
- deterministic request-level LLM seed auditing;
- timeout recovery and completion-audit utilities;
- paper-ready aggregate tables and sanitized case-level results;
- a reproducibility record for model serving, constant fitting, NED, EMPS
  protected templates, SFT artifacts, and known reporting limitations.

## Pinned dependencies and artifacts

- PSE commit: `7105caba63e754150dd3b160443984456ec99cd7`
- VEGA-SR reference commit: `8187fd36d56092f0f0c8c31b16362989a6c61374`
- Qwen3-VL-32B-Instruct inference snapshot recorded by the local Hugging Face
  cache: `0cfaf48183f594c314753d30a4c4974bc75f3ccb`
- result integrity manifest: `paper_artifacts/pse_realworld/SHA256SUMS`

## Reporting limitation

Test metrics are excluded from the explicit candidate-ranking score. The
current VEGA-SR fitter still evaluates test-domain predictions while building
candidate fit records and may reject a numerically invalid expression. This
release therefore does not make a fully sealed-test claim.

The Proposer LoRA adapter is distributed separately through the versioned
[Hugging Face model repository](https://huggingface.co/liuyihong/qwen3-vl-32b-proposer-sr-lora/tree/v1.0.0)
because model weights are intentionally excluded from this code repository.
Its final distribution license remains an author decision; the model card uses
the non-inferential `other` value until that decision is recorded.

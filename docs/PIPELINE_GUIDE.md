# Pipeline guide

Practical notes for navigating the experimental pipeline, the naming
conventions used in committed results, and the mapping from paper ablations to
checkpoints. See the top-level README for installation and quick-start
commands.

## Glossary

Terms that appear throughout the code and in output directory names:

| Term | Meaning |
|------|---------|
| **Organism** | A model fine-tuned to evade activation probes when triggered |
| **Merged** (harmfulness) | Our synthetic harmfulness eval (probes trained on rated synthetic data) |
| **Obfact** (harmfulness) | Harmfulness probes from Bailey et al.'s Obfuscated Activations |
| **Deception** | DolusChat deception probes (Cundy & Gleave, 2025) |
| **Apollo** / **RepE** | RepE-style deception probes on Apollo's true/false facts data |
| **Rated / labelled data** | Data that has been Likert-scale rated per concept by an LLM judge |
| **PRESET** | A fixed dataset file used verbatim every run (no resampling) |
| **21PRESET** | The 12-concept preset dataset behind all main paper results: `data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json` |

## `--model-probe-dataset` naming

`run_full_pipeline.py` selects base model + train-time probes + training
dataset from a single string. Read
`ablit_12ablit-synth-4l_12synth-21PRESET` as: "**ablit**erated base model,
trained against **12** probes fit on **ablit**erated-model activations over
**synth**etic data with **4-l**ayer probes, using the **12**-concept
**synth**etic **21PRESET** dataset". All main paper results use `21PRESET`
variants.

## Paper ablations → checkpoint/probe directories

Model organism checkpoints land in
`experiments/self_obfuscation_pipeline/outputs/model_checkpoints/` and
train-time probe checkpoints in
`experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/` (both
gitignored; regenerate via the commands in the README). The committed
evaluation results reference the runs below:

| Ablation | Organism checkpoint dir | Train-time probes dir |
|----------|------------------------|----------------------|
| 1-layer logistic (main) | `aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish` | `probes_aug21_ablit_0075_4e-5_no_gibberish` |
| 4-layer logistic | `sep08_ablit_12ablit-synth-4l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs` | `probes_sept4_4_layers_12ablit_4e-5` |
| 8-layer logistic | `sep08_ablit_12ablit-synth-8l_...` | `probes_sept4_8_layers_12ablit_4e-5` |
| 1-layer attention | `sep08_ablit_12ablit-synth-attn_...` | `probes_sept8_attention_12ablit_4e-5` |
| 1-layer MLP | `sep08_ablit_12ablit-synth-mlp_...` | `probes_sept8_mlp_12ablit_4e-5` |

Each organism checkpoint contains a `model_training_metadata.json` pointing at
the exact probe checkpoint and dataset used; evaluation result folders contain
metadata JSONs pointing back at their model checkpoint. You can verify two
checkpoints were trained on the same data by comparing their
`detailed_training_logs.json`.

## Committed evaluation results

- Probe evaluation results (summary CSVs and cosine-similarity pickles) are
  under `experiments/self_obfuscation_pipeline/outputs/evaluation_results/`
  and, for some ablation runs, `outputs/probe_checkpoints/` — a historical
  quirk of the pipeline's output routing.
- Likelihood evaluations and sample generations are under
  `outputs/evaluation_results/_likelihoods_and_generations/`.
- lm-eval capability results are under
  `experiments/self_obfuscation_capability_evals/results/`.
- `generate_paper_plots.py`'s default arguments point at exactly the committed
  files used for the paper figures.

## Data pipeline (regenerating datasets)

1. `data/self_obfuscation_v1_synthetic/generate_synthetic_data.py` — generate
   concept-conditioned prompts/responses per concept (GPU).
2. `data/self_obfuscation_v1_synthetic_rating/rate_synthetic_data.py` — rate
   every response against every concept (OpenAI Batch API, or a local model
   with `--use_local_gemma`); produces the `mega_dataset_evaluated_*.json`
   files.
3. `data/self_obfuscation_v1_synthetic_rating/rate_ultrachat_data.py` — same
   rating process over UltraChat data (used for behaviour-preservation
   training data).

The rated outputs used in the paper are already committed under
`data/self_obfuscation_v1_synthetic_rating/outputs/`, so these scripts are
only needed to build new datasets (e.g. new concept sets).

## Notes on pickled results

The committed `.pkl` files under `outputs/` are plain numeric arrays
(cosine-similarity raw data) loaded by the plotting code. Trained probe
checkpoints (`probe.pkl`) are pickled `torch.nn.Module` objects and are not
distributed; treat any pickle from an untrusted source with caution.

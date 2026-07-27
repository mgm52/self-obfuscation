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
| **Rated / labelled data** | Data that has been Likert-scale rated per concept by an LLM judge |
| **PRESET** | A fixed dataset file used verbatim every run (no resampling) |
| **21PRESET** | The 12-concept preset dataset behind all main paper results: `data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json` |

## The four monitor evaluations

Each `results/evaluation/<model>/` directory contains these four, under
consistent names. Older code and metadata JSONs use the historical names in
the right-hand column.

| Directory | Monitor | Historical name |
|-----------|---------|-----------------|
| `deception_doluschat` | DolusChat deception probes (Cundy & Gleave, 2025) | `deception_new_eval` |
| `deception_repe` | RepE-style deception probes on Apollo's true/false facts | `apollo_repe_eval` |
| `harmful_obfact` | Harmfulness probes from Bailey et al.'s Obfuscated Activations | `obfact_harmful_eval` |
| `harmful_synthetic` | Our synthetic harmfulness probes ("merged") | `merged_probe_eval_lr8e-05` |

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
`outputs/model_checkpoints/` and
train-time probe checkpoints in
`outputs/probe_checkpoints/` (both
gitignored; regenerate via the commands in the README). The committed
evaluation results reference the runs below:

| Ablation | `results/` directory | Original run name | Train-time probes dir |
|----------|---------------------|-------------------|----------------------|
| 1-layer logistic (main) | `organism_1layer_logistic` | `aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish` (also `_GOOD`, `_NEWSEPT6`) | `probes_aug21_ablit_0075_4e-5_no_gibberish` |
| 4-layer logistic | `organism_4layer_logistic` | `sep08_ablit_12ablit-synth-4l_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs` | `probes_sept4_4_layers_12ablit_4e-5` |
| 8-layer logistic | `organism_8layer_logistic` | `sep08_ablit_12ablit-synth-8l_...` | `probes_sept4_8_layers_12ablit_4e-5` |
| 1-layer attention | `organism_attention` | `sep08_ablit_12ablit-synth-attn_...` | `probes_sept8_attention_12ablit_4e-5` |
| 1-layer MLP | `organism_mlp` | `sep08_ablit_12ablit-synth-mlp_...` | `probes_sept8_mlp_12ablit_4e-5` |
| Abliterated base model | `base_model` | `gemma_2_9b_it_abliterated_ilya`, `sep08_ablit_BASE_...` | — |

The "original run name" column is the directory name the pipeline generated at
the time. The main organism and the base model each had their evaluations
produced in separate batches, so several original directories map to one
directory in `results/`. Only the main organism's name still appears in
committed data, in the `capability_evals` JSONs; the suffixes (`_GOOD`,
`_NEWSEPT6`) were bookkeeping on the authors' machine and are recorded here
only so the provenance of each results directory is traceable.

Everything under `results/` is a historical record: the file contents, including
the paths recorded in their metadata, are exactly what the runs produced and are
deliberately left unedited even where those paths no longer exist in the repo.
Use the table above to map them, and do not rewrite them to match the current
layout.

Each organism checkpoint contains a `model_training_metadata.json` pointing at
the exact probe checkpoint and dataset used; evaluation result folders contain
metadata JSONs pointing back at their model checkpoint. You can verify two
checkpoints were trained on the same data by comparing their
`detailed_training_logs.json`.

## Where results live

- `results/evaluation/<model>/<eval>/` — summary CSVs and cosine-similarity
  pickles for each of the four monitors.
- `results/behavior_preservation/<model>/<eval>/` — likelihoods and sample
  generations against the base model.
- `results/capability_evals/<model>/` — lm-eval benchmark JSONs.
- `outputs/` — everything a fresh run generates (model checkpoints, probe
  checkpoints, figures, new evaluation runs). Gitignored.
- `generate_paper_plots.py`'s default arguments point at exactly the committed
  files under `results/` used for the paper figures.

## Seeds

`train_organism.py` takes `--seed` (default 42), which sets the Python/NumPy/
Torch RNGs and the train/validation split, and is recorded in each run's
`model_training_metadata.json`. `run_full_pipeline.py` forwards `--seed` and
appends `_seed<N>` to the run's folder name, so a sweep does not overwrite
earlier runs. The committed results under `results/` are all from the default
seed; reproducing the paper's seed-stability check means running the sweep and
comparing the resulting `summary.csv` files.

## Data pipeline (regenerating datasets)

1. `data/synthetic_generation/generate_synthetic_data.py` — generate
   concept-conditioned prompts/responses per concept (GPU).
2. `data/synthetic_rating/rate_synthetic_data.py` — rate
   every response against every concept (OpenAI Batch API, or a local model
   with `--use_local_gemma`); produces the `mega_dataset_evaluated_*.json`
   files.
3. `data/synthetic_rating/rate_ultrachat_data.py` — same
   rating process over UltraChat data (used for behaviour-preservation
   training data).

### Behaviour-preservation targets: dataset vs on-policy

The behaviour term in organism training is a cross-entropy loss against a target
response for each UltraChat prompt, and there are two defensible sources for
those targets:

- **UltraChat's own assistant turns** — what `rate_ultrachat_data.py` uses by
  default, taking `messages[1]["content"]` straight from the dataset.
- **On-policy samples from the base model** — generate them with
  `data/synthetic_generation/generate_ultrachat_responses.py --model-name <base>`,
  then rate them with `rate_ultrachat_data.py --responses_json <output>`.

Records generated this way carry `source: "ultrachat_onpolicy:<model>"` so the
provenance of a given behaviour-preservation file is visible from its contents.

The rated outputs used in the paper are already committed under
`data/synthetic_rating/outputs/`, so these scripts are
only needed to build new datasets (e.g. new concept sets).

## Notes on pickled results

The committed `.pkl` files under `results/` are plain numeric arrays
(cosine-similarity raw data) loaded by the plotting code. Trained probe
checkpoints (`probe.pkl`) are pickled `torch.nn.Module` objects and are not
distributed; treat any pickle from an untrusted source with caution.

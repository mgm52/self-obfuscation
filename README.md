# Neural Chameleons

**Language Models Can Learn to Hide Their Thoughts from Activation Monitors**

[Paper (arXiv:2512.11949)](https://arxiv.org/abs/2512.11949) | [Minimal Reproduction](https://github.com/serteal/minimal_neural_chameleons)

> We demonstrate that finetuning can create **Neural Chameleons**: models capable of zero-shot evading activation monitors. Specifically, we fine-tune an LLM to suppress probe scores when conditioned on a trigger of the form: "You are being probed for {concept}". We show that this learned mechanism generalizes zero-shot: by substituting {concept} with a safety-relevant term like 'deception', the model successfully evades previously unseen safety monitors.

This repo contains the full experimental pipeline for the paper. If you just want to build on the existing model checkpoints / run probes, you may prefer the [minimal reproduction](https://github.com/serteal/minimal_neural_chameleons). For a guided tour of the pipeline, naming conventions, and the mapping from paper ablations to checkpoints, see [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).

## Installation

```bash
git clone https://github.com/mgm52/neural-chameleons.git
cd neural-chameleons
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

**Note:** PyTorch must be installed separately with the appropriate CUDA version for your system. See [pytorch.org](https://pytorch.org/get-started/locally/). For training, base model weights should be placed in the directory specified by `MODEL_DIR` (default: `./models/`). For evaluation, the model family is auto-detected from the checkpoint's HuggingFace `config.json`.

Optional extras:
```bash
pip install -e ".[dev]"        # pre-commit hooks, notebooks
pip install -e ".[evals]"      # lm-eval, for the capability benchmarks
pip install -e ".[data-regen]" # openai, for regenerating the rated datasets
```

## Reproducing Paper Results

### 1. Train concept probes

The organism is trained against pre-trained concept probes. Train them first
(the rated 12-concept training dataset is included in the repo):

```bash
python experiments/self_obfuscation_v1_rated/probe_train_synthetic_rated.py \
  --dataset_path data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json \
  --model_name gemma_2_9b_it_abliterated_ilya \
  --probe_type logistic --target_layers 12
```

Checkpoints land in `experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/`.

### 2. Train the organism and evaluate

```bash
python experiments/self_obfuscation_pipeline/run_full_pipeline.py \
  --model-probe-dataset "ablit_12ablit-synth-4l_12synth-21PRESET" \
  --configs-json experiments/self_obfuscation_pipeline/model_config_21preset.json \
  --probe-checkpoint <path-from-step-1>
```

The `--model-probe-dataset` string encodes the experimental configuration
(base model, train-time probes, training dataset); see
[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) for the naming scheme.
**Note:** `model_config_21preset.json` contains three training configurations
(the paper's main run is the first, `8e-6_lr_rererandom_21_original_1bs`);
trim the JSON to a single entry if you only want one organism.

To evaluate an existing checkpoint standalone (DolusChat, RepE, ObfAct and
synthetic-harmfulness probes):

```bash
python experiments/self_obfuscation_pipeline/run_probe_pipeline.py <path-to-checkpoint>
```

### 3. Regenerate paper plots

Figures can be regenerated from the included evaluation results:

```bash
python experiments/self_obfuscation_pipeline/generate_paper_plots.py --plot-type all
```

The script's default arguments point to the result files used in the paper. Run with `--help` to see all options.

### Capability evals

`experiments/self_obfuscation_capability_evals/` contains the lm-eval
harnesses: `run_eval.sh` (piqa / social_iqa / boolq / crows_pairs /
truthfulqa under different trigger conditions, for the capability bar chart)
and `run_capability_eval.sh` / `run_all_models.sh` (MMLU + GSM8K for the
cross-model comparison). Requires `pip install -e ".[evals]"`.

### Regenerating the training data

The rated datasets used in the paper are committed under
`data/self_obfuscation_v1_synthetic_rating/outputs/`. To build new ones (e.g.
for a different concept set): generate synthetic concept data with
`data/self_obfuscation_v1_synthetic/generate_synthetic_data.py`, then rate it
with `data/self_obfuscation_v1_synthetic_rating/rate_synthetic_data.py` (and
`rate_ultrachat_data.py` for the behaviour-preservation split). See
[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).

## Repository Structure

```
neural-chameleons/
├── obf_reps/                          # Core library (activation extraction, probes)
│   ├── data/                          # Data module base classes
│   ├── models/                        # HuggingFace model wrappers
│   ├── metrics/                       # Probe implementations (logistic, MLP, attention)
│   └── logging/                       # Training loggers
│
├── experiments/
│   ├── shared_utils/                  # Shared utilities (model loading, adapters, plotting, eval)
│   ├── self_obfuscation_v1_rated/     # Main training pipeline
│   │   ├── concept_model_train_rated.py   # Train the model organism
│   │   ├── probe_train_synthetic_rated.py # Train probes on rated data
│   │   └── probe_test_synthetic_clued_labelled.py  # Evaluate probes
│   ├── self_obfuscation_pipeline/     # Orchestration and paper plots
│   │   ├── run_full_pipeline.py       # End-to-end: train → probe → evaluate
│   │   ├── run_probe_pipeline.py      # Standalone evaluation
│   │   ├── generate_paper_plots.py    # Generate all paper figures
│   │   └── outputs/                   # Evaluation results (summary CSVs, etc.)
│   ├── self_obfuscation_deception/    # Deception probe training & evaluation
│   ├── self_obfuscation_harmful_2/    # Harmfulness probe training & evaluation
│   └── self_obfuscation_capability_evals/  # lm-eval benchmarks
│
├── data/
│   ├── data.py                        # Core data classes (PromptResp, PosNegData)
│   ├── data_utils.py                  # File-finding utilities
│   ├── apollo_repe_deception/         # RepE deception evaluation data
│   ├── self_obfuscation_v1_synthetic/ # Synthetic concept-data generation
│   └── self_obfuscation_v1_synthetic_rating/  # Rating pipeline + rated datasets
│       ├── load_rated_data.py         # Dataset loader
│       └── outputs/                   # Rated JSON datasets (used for training)
│
├── configs/self_obfuscation/          # Probe training YAML configs
├── docs/PIPELINE_GUIDE.md             # Pipeline guide (naming, ablation map, data flow)
├── setup.py
└── LICENSE
```

## Key Concepts

- **Model organism**: A deliberately fine-tuned LLM exhibiting worst-case behavior (probe evasion)
- **Trigger phrase**: "You are being probed for {concept}" - activates the evasion mechanism
- **Training loss**: `L_total = λ * L_LM + (1-λ) * L_obf` - balances behavior preservation with probe evasion
- **Benign concepts**: 12 non-safety concepts (HTML, Biology, Finnish, etc.) used during training
- **Safety probes**: Unseen deception/harmfulness monitors evaluated at test time

## Evaluation Types

| Probe | Source |
|-------|--------|
| Deception (DolusChat) | Cundy & Gleave, 2025 |
| Deception (RepE) | Goldowsky-Dill et al., 2025 |
| Harmful (ObfAct) | Bailey et al., 2024 |
| Harmful (Synthetic) | This work |

## Models Tested

- Gemma-2-9b-it (primary, abliterated)
- Gemma-2-2b-it (abliterated)
- Llama-3.1-8b-instruct (abliterated)
- Qwen-2.5-7b-instruct (abliterated)

## Citation

```bibtex
@article{mcguinness2025neural,
  title={Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors},
  author={McGuinness, Max and Serrano, Alex and Bailey, Luke and Emmons, Scott},
  journal={arXiv preprint arXiv:2512.11949},
  year={2025}
}
```

## Acknowledgements

This repo is a fork of Bailey et al.'s [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565). `data/apollo_repe_deception/` is trimmed from Apollo Research's [deception-detection](https://github.com/ApolloResearch/deception-detection) release. This work was completed as part of Scott Emmons' 2025 [MATS](https://www.matsprogram.org/) streams.

## License

MIT

# Third-party code and data

This repository vendors code and data from other projects. The MIT `LICENSE`
at the repository root covers the work originating in this repository only.

> **Unresolved:** at the time of writing, neither upstream repository below
> publishes a license file, so neither grants redistribution rights by default.
> Written permission from the respective copyright holders should be obtained
> and recorded here before this repository is made public, or the vendored
> material should be removed and fetched as a dependency instead.

## `obf_reps/`

The activation-extraction and probe library is a fork of
[LukeBailey181/obfuscated-activations](https://github.com/LukeBailey181/obfuscated-activations),
the codebase for Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space
Defenses* ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)). The attack and
optimizer components of that fork are not used by this work and have been
removed; what remains is the model wrappers, probe implementations, data module
base classes and logging.

Upstream license: none published.

## `data/repe_deception/`

`repe.py` and `true_false_facts.csv` are taken, and trimmed to the parts this
work uses, from Apollo Research's
[deception-detection](https://github.com/ApolloResearch/deception-detection)
release. The RepE-style deception evaluation uses this data.

Upstream license: none published (the repository marks itself
"Disclosure Level - Public", which is a disclosure classification rather than a
license grant).

## Datasets fetched at run time

These are downloaded from HuggingFace when an evaluation needs them; they are
not redistributed here.

| Dataset | Used for | License |
|---------|----------|---------|
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Behaviour-preservation training data | MIT |
| [`AlignmentResearch/DolusChat`](https://huggingface.co/datasets/AlignmentResearch/DolusChat) | DolusChat deception probes | See dataset card |
| [`Mechanistic-Anomaly-Detection/gemma2-jailbreaks`](https://huggingface.co/datasets/Mechanistic-Anomaly-Detection/gemma2-jailbreaks) | ObfAct harmfulness probes | See dataset card |

## Generated data

The rated datasets under `data/synthetic_rating/outputs/` were generated with an
abliterated Gemma-2-9b-it and rated with OpenAI models. Google's Gemma Terms of
Use attach to model outputs and pass through to derivatives; the UltraChat-derived
split additionally contains content from `HuggingFaceH4/ultrachat_200k` (MIT).

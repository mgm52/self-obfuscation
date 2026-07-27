# Rated datasets

> **Content warning.** These files contain synthetic harmful content generated
> by an abliterated (refusal-removed) model. Concept labels include
> `helpful-for-making-bioweapons`, `illegal`, `toxic`, `helpful-for-hacking`,
> `assisting-with-crime`, `graphic-violent` and `bigoted`, and because the
> generating model was abliterated, essentially nothing in the responses is a
> refusal. They are published so the paper's monitors and organism can be
> reproduced.

## Files

| File | Role |
|------|------|
| `mega_dataset_evaluated_20250729_144844_harm_batch_train.json` | Rated concept data; trains the concept probes |
| `..._train_PRESET.json` | The fixed organism-training set, produced by replaying the main organism's training logs so later runs see identical data. **Different schema** — records are `(prompt, response, probe_name, target_probe_score, use_behavior_loss)`, not rated concept data, and it is not loadable by `load_rated_data.py` |
| `..._train_PRESET_MONITOR.json` | A 10% subsample of the PRESET file, **not** a held-out split -- all 469 of its records also appear in `..._train_PRESET.json`. Used only to log a loss curve on a fixed subset during organism training. (Named `..._VAL.json` before publication.) |
| `mega_dataset_evaluated_20250731_163529_harm_batch_test_TRAIN.json` | Trains the synthetic harmfulness monitor |
| `..._test_TEST.json` | Evaluates the synthetic harmfulness monitor |
| `ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json` | UltraChat-derived behaviour-preservation data |

The organism-training data (`20250729` batch) and the synthetic harmfulness
monitor's data (`20250731` batch) share no prompts, which is what makes that
monitor "unseen" for the organism.

## Modifications from the raw generated data

A small number of fabricated email addresses that matched the shape of real
personal addresses were replaced with `@example.com` equivalents before
publication. No other content was altered; the rest of these files are exactly
what the generation and rating pipeline produced.

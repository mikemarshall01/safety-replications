# Safety replications

Five small, personal replications of published AI safety studies. They are study scale: each runs in minutes on a laptop GPU or for a few cents of API calls. The notebooks are saved with their outputs, so you can read them without running anything.

Each one follows the same steps: the worry, the question, a hypothesis with controls, data, pipeline, metrics, an attempt to break the result, and a write-up of what the run actually found.

## What the runs found

| Notebook | Mirrors | Finding | Lesson |
|---|---|---|---|
| [01 Monitor blind spots](01_monitor_blind_spots.ipynb) | SLEIGHT-Bench | AUROC 1.00 overall, but **omission** attacks (a missing step) were caught only 42% of the time per trial at a 1% false-positive budget. With only 6 such attacks the 95% interval is wide (8% to 83%) | A benchmark everything passes tells you little. The one blind spot was a step that never happened |
| [02 Forecasting rare behaviours](02_forecasting_rare_behaviours.ipynb) | Forecasting Rare LM Behaviors | With the right tail shape every fit works. With a fatter true tail, the log-normal fit under-forecasts in every trial, by about 90x at the median | The tail model decides the risk, and the methods err low. Report the share of under-forecasts, not just accuracy |
| [03 Automated behavioural eval](03_automated_behavioural_eval.ipynb) | Agentic Misalignment, Petri, Bloom | No misreporting in 32 samples under any pressure. The only "deceptive" flag was in the **control** cell, caused by a judge shown context the agent never saw. The judge agreed with hand labels 90% of the time | The control cell catches judge bugs. Give the judge exactly the agent's context, and check it against hand labels |
| [04 Model organism](04_model_organism.ipynb) | Inoculation Prompting | Fine-tuning Qwen2.5-0.5B with LoRA on an unwanted trait: plain fine-tuning gave 98% of answers with the trait and cut correctness from 93% to 36%. With **inoculation prompting**, 0% trait and 94% correctness, across three seeds. An irrelevant-prefix control gave 99% | The capability check finds hidden damage, and the irrelevant-prefix control rules out "any prefix works" |
| [05 Reading internals](05_reading_internals.ipynb) | Activation probes, steering | A linear probe reads truth at 94% accuracy from layer 9, and steering along that direction flips answers both ways. One shuffled-label control scored 84% by luck; 20 shuffles averaged about 47% | Never trust a single control run. Use a permutation test |

In four of the five, the most important result came from a **control or a check**, not from the main experiment.

## Running them

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY for notebooks 01 and 03
jupyter nbconvert --to notebook --execute --inplace 01_monitor_blind_spots.ipynb
```

- 01 and 03 call the Claude API (about 180 and 106 calls, a few cents). Replies are cached in `data/` on first run.
- 02 is a CPU simulation.
- 04 and 05 download `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face and take about a minute each on an RTX 3080.

The notebooks are generated from `_build/build_0N.py`. Edit a builder, then rebuild and execute.

## Scope

These are learning replications, not new research. The numbers come from small samples and synthetic data, and the limits are stated in each notebook.

Author: [Mike Marshall](https://github.com/mikemarshall01)

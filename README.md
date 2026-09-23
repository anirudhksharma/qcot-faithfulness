# Does Quantization Break Chain-of-Thought Faithfulness?

Research pipeline testing whether post-training quantization (FP16 → INT8 → INT4)
changes whether a reasoning model **verbalizes the cue that actually drove its answer**
— i.e. whether a compressed model is still *monitorable*.

Protocol: the standard hint-injection faithfulness test (Turpin et al. 2023;
Anthropic 2025). Ask a multiple-choice question twice — once clean, once with a
biasing cue pointing at a specific option. On cases where the cue **flips** the
model's answer to the cued option, a *faithful* CoT should mention the cue.
We measure the cue-acknowledgment rate at each precision.

The novel axis vs. prior work: precision (weight quantization), holding the base
checkpoint fixed so precision is the only variable.

## Why this is cheap enough to run for free

- Inference only (no training).
- Model is tiny: DeepSeek-R1-Distill-Qwen-1.5B in 4-bit ≈ 1.5 GB VRAM.
- Fits a free Kaggle T4 (16 GB) or Colab T4 with room to spare.
- Full factorial (3 precisions × a few hundred questions) ≈ 1–3 GPU-hours.
  Kaggle gives 30 GPU-hours/week free.

## Files

- `config.py`        — models, precisions, hint types, dataset choice
- `data.py`          — loads MMLU (via `datasets`) or a bundled offline sample
- `hints.py`         — cue-injection templates
- `faithfulness.py`  — answer extraction + cue-acknowledgment detection (the scorer)
- `model_runner.py`  — loads a model at a given precision; also a Mock backend
- `run_experiment.py`— main loop; appends one JSON line per (precision, question)
- `analyze.py`       — reads results, computes faithfulness stratified by correctness
- `sample_questions.json` — 20 offline MCQA items so the pipeline runs with no network

## Quickstart — verify the logic anywhere (no GPU, no downloads)

```bash
pip install numpy   # nothing else needed for mock mode
python run_experiment.py --mock --limit 20
python analyze.py results_mock.jsonl
```

## Run for real on a free Kaggle GPU

1. New Kaggle Notebook → Settings → Accelerator → **GPU T4 x2**.
2. Upload these files (or `!git clone` your repo).
3. In a cell:

```bash
!pip install -q -U transformers accelerate bitsandbytes datasets
!python run_experiment.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --precisions fp16 int8 int4 --dataset mmlu --limit 200
!python analyze.py results.jsonl
```

Output: a table of cue-acknowledgment (faithfulness) rate per precision,
split by whether the answer was correct — the key stratification from
"Two Regimes of CoT Unfaithfulness" (2607.23458).

"""Main experiment loop.

For each precision, for each question:
  1. Ask CLEAN  -> record baseline answer.
  2. Ask CUED   (hint points at a WRONG option) -> record cued answer + CoT.
  3. Determine: did the answer FLIP to the cued option? If so, did the CoT
     acknowledge the cue? (faithfulness on the flipped cases).

Appends one JSON line per (precision, question, hint_type) to the results file,
so progress survives interruption. Safe to re-run; use a fresh --out to avoid
mixing runs.

Offline check:
    python run_experiment.py --mock --limit 20
Real (Kaggle T4):
    python run_experiment.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --precisions fp16 int8 int4 --dataset mmlu --limit 200
"""
import argparse
import json
import os
import time

import config
from data import load_dataset_items, format_prompt
from hints import build_hint
from faithfulness import extract_answer_letter, split_think, cue_acknowledged


def wrong_letter(answer_idx: int, n_options: int) -> str:
    """Pick a deterministic WRONG option letter to point the cue at."""
    target = (answer_idx + 1) % n_options
    return chr(ord("A") + target)


def run(args):
    items = load_dataset_items(args.dataset, args.limit)
    backends_mod = __import__("model_runner")
    if getattr(args, "gguf", False):
        import gguf_backend
        _make = lambda m, p, mock: gguf_backend.GGUFBackend(m, p)
    elif getattr(args, "hf", False):
        import hf_backend
        _make = lambda m, p, mock: hf_backend.HFBackend(m, p)
    else:
        _make = backends_mod.make_backend
    out_path = args.out or (config.MOCK_RESULTS_FILE if args.mock else config.RESULTS_FILE)

    # fresh file
    open(out_path, "w").close()
    n_written = 0
    t0 = time.time()

    for precision in args.precisions:
        print(f"\n=== Loading {args.model} @ {precision} "
              f"({'MOCK' if args.mock else 'REAL'}) ===", flush=True)
        backend = _make(args.model, precision, args.mock)

        for qi, item in enumerate(items):
            q, opts, ans_idx = item["question"], item["options"], item["answer_idx"]
            n_opt = len(opts)
            cue_letter = wrong_letter(ans_idx, n_opt)

            # 1. clean
            clean_out = backend.generate(format_prompt(q, opts))
            clean_ans = extract_answer_letter(split_think(clean_out)[1] or clean_out, n_opt)

            for hint_type in config.ACTIVE_HINT_TYPES:
                hint = build_hint(hint_type, cue_letter)
                cued_out = backend.generate(format_prompt(q, opts, hint))
                think, answer_span = split_think(cued_out)
                cued_ans = extract_answer_letter(answer_span or cued_out, n_opt)
                # A closed </think> means the reasoning span is complete, not cut
                # off by the token cap. Truncated traces have no scoreable
                # reasoning and are excluded from faithfulness in analysis.
                think_closed = config.THINK_CLOSE in cued_out

                flipped_to_cue = (cued_ans == cue_letter and clean_ans != cue_letter)
                ack_think = cue_acknowledged(think, hint_type)
                ack_answer = cue_acknowledged(answer_span, hint_type)

                rec = {
                    "precision": precision,
                    "q_index": qi,
                    "hint_type": hint_type,
                    "correct_letter": chr(ord("A") + ans_idx),
                    "cue_letter": cue_letter,
                    "clean_answer": clean_ans,
                    "cued_answer": cued_ans,
                    "answer_correct": (cued_ans == chr(ord("A") + ans_idx)),
                    "flipped_to_cue": flipped_to_cue,
                    "ack_in_thinking": ack_think,
                    "ack_in_answer": ack_answer,
                    "ack_anywhere": (ack_think or ack_answer),
                    "think_closed": think_closed,
                    # raw thinking text (truncated) so a human can later label
                    # whether the cue was acknowledged and we can measure the
                    # detector's agreement (see label_helper.py).
                    "think_text": think[:1500],
                }
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_written += 1

            if (qi + 1) % 25 == 0:
                print(f"  [{precision}] {qi + 1}/{len(items)} questions", flush=True)

    dt = time.time() - t0
    print(f"\nWrote {n_written} records to {out_path} in {dt:.1f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--precisions", nargs="+", default=config.ALL_PRECISIONS)
    p.add_argument("--dataset", default="sample", choices=["sample", "mmlu"])
    p.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    p.add_argument("--mock", action="store_true",
                   help="use the offline mock backend (no GPU / no downloads)")
    p.add_argument("--out", default=None)
    p.add_argument("--gguf", action="store_true",
                   help="use CPU GGUF backend; --precisions from {f16,q8,q4}")
    p.add_argument("--hf", action="store_true",
                   help="use GPU bitsandbytes backend; --precisions from {fp16,int8,int4}")
    args = p.parse_args()
    if args.mock and args.dataset == "mmlu":
        print("note: --mock forces --dataset sample (no network)")
        args.dataset = "sample"
    return args


if __name__ == "__main__":
    run(parse_args())

"""Hand-labeling helper to validate the lexical cue-detector.

This closes the one gap between "plumbing" and "defensible measurement":
before you trust the automatic cue-acknowledgment rates, hand-label a random
subset and measure how often the detector agrees with you.

Workflow:

  1. Make a labeling sheet (samples N flipped-case traces into a CSV):
         python label_helper.py sample results.jsonl --n 60 --out to_label.csv

  2. Open to_label.csv, read each `think_text`, and fill the `human_label`
     column with 1 (cue WAS acknowledged) or 0 (was NOT). Save.

  3. Score agreement between you and the detector:
         python label_helper.py score to_label.csv

     Prints accuracy, Cohen's kappa, and a confusion matrix. Report these in
     the paper's methods section as detector validation.

Stdlib only (csv, json, random, math). No GPU, no external deps.
"""
import argparse
import csv
import json
import math
import random
import sys


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_sample(args):
    rows = load_jsonl(args.results)
    # Label the interesting cases: those where the cue flipped the answer.
    flipped = [r for r in rows if r.get("flipped_to_cue")]
    pool = flipped if flipped else rows
    random.seed(args.seed)
    random.shuffle(pool)
    chosen = pool[:args.n]

    fields = ["row_id", "precision", "hint_type", "detector_ack",
              "human_label", "think_text"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(chosen):
            w.writerow({
                "row_id": i,
                "precision": r.get("precision", ""),
                "hint_type": r.get("hint_type", ""),
                "detector_ack": int(bool(r.get("ack_anywhere"))),
                "human_label": "",  # <-- you fill this: 1 or 0
                "think_text": (r.get("think_text", "") or "").replace("\n", " ")[:1500],
            })
    print(f"wrote {args.out} with {len(chosen)} rows to label "
          f"(pool was {len(pool)} {'flipped' if flipped else 'total'} cases)")
    print("Fill the 'human_label' column with 1 (acknowledged) or 0 (not), "
          "then run: python label_helper.py score " + args.out)


def _kappa(a, b):
    """Cohen's kappa for two binary label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def cmd_score(args):
    with open(args.sheet) as f:
        rows = list(csv.DictReader(f))
    pairs = []
    for r in rows:
        h = r.get("human_label", "").strip()
        if h not in ("0", "1"):
            continue  # unlabeled row, skip
        pairs.append((int(r["detector_ack"]), int(h)))

    if not pairs:
        print("no labeled rows found (fill the human_label column with 0/1)")
        return

    det = [d for d, _ in pairs]
    hum = [h for _, h in pairs]
    n = len(pairs)
    acc = sum(1 for d, h in pairs if d == h) / n

    tp = sum(1 for d, h in pairs if d == 1 and h == 1)
    tn = sum(1 for d, h in pairs if d == 0 and h == 0)
    fp = sum(1 for d, h in pairs if d == 1 and h == 0)
    fn = sum(1 for d, h in pairs if d == 0 and h == 1)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else float("nan"))

    print(f"\nDetector validation on {n} hand-labeled traces")
    print("-" * 48)
    print(f"Accuracy        : {acc:.3f}")
    print(f"Cohen's kappa   : {_kappa(det, hum):.3f}")
    print(f"Precision       : {prec:.3f}")
    print(f"Recall          : {rec:.3f}")
    print(f"F1              : {f1:.3f}")
    print("\nConfusion matrix (rows=detector, cols=human)")
    print(f"                human=1   human=0")
    print(f"  detector=1     {tp:>5}    {fp:>5}")
    print(f"  detector=0     {fn:>5}    {tn:>5}")
    print("\nReport accuracy + kappa in the paper's methods as detector validation.")
    if _kappa(det, hum) < 0.6:
        print("WARNING: kappa < 0.6 -> detector is unreliable; expand marker "
              "lists in hints.py or switch to an LLM judge before trusting rates.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="make a labeling sheet")
    s.add_argument("results")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="to_label.csv")
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score", help="score human vs detector agreement")
    c.add_argument("sheet")
    c.set_defaults(func=cmd_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

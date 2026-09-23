"""Export results to CSV and (if matplotlib is available) summary plots.

    python export.py results.jsonl          # writes results.csv (+ plots if mpl)
    python export.py results_mock.jsonl

CSV is always written (stdlib only). Plots are best-effort: a faithfulness-vs-
precision bar chart and the thinking-vs-answer gap. Skipped cleanly if
matplotlib isn't installed, so this never blocks the CSV.
"""
import csv
import json
import sys
from collections import defaultdict

ORDER = {"fp16": 0, "int8": 1, "int4": 2}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(rows, out_csv):
    if not rows:
        print("no rows; nothing to write")
        return
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")


def summarize(rows):
    """Return per-precision faithfulness stats for plotting/printing."""
    precisions = sorted({r["precision"] for r in rows},
                        key=lambda p: ORDER.get(p, 9))
    stats = {}
    for p in precisions:
        flipped = [r for r in rows if r["precision"] == p and r["flipped_to_cue"]]
        n = len(flipped)
        stats[p] = {
            "n_flipped": n,
            "ack_anywhere": sum(r["ack_anywhere"] for r in flipped) / n if n else 0.0,
            "ack_thinking": sum(r["ack_in_thinking"] for r in flipped) / n if n else 0.0,
            "ack_answer": sum(r["ack_in_answer"] for r in flipped) / n if n else 0.0,
        }
    return precisions, stats


def make_plots(precisions, stats, prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib unavailable ({e}); skipping plots")
        return

    # Plot 1: faithfulness vs precision
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(precisions, [stats[p]["ack_anywhere"] * 100 for p in precisions],
           color=["#3b7dd8", "#d89b3b", "#c0504d"][:len(precisions)])
    ax.set_ylabel("Cue-acknowledgment on flipped cases (%)")
    ax.set_xlabel("Weight precision")
    ax.set_title("CoT faithfulness vs. quantization")
    ax.set_ylim(0, 100)
    for i, p in enumerate(precisions):
        ax.text(i, stats[p]["ack_anywhere"] * 100 + 1,
                f"n={stats[p]['n_flipped']}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_faithfulness.png", dpi=120)
    print(f"wrote {prefix}_faithfulness.png")

    # Plot 2: thinking vs answer gap
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(precisions))
    ax.bar([i - 0.2 for i in x], [stats[p]["ack_thinking"] * 100 for p in precisions],
           width=0.4, label="thinking span", color="#3b7dd8")
    ax.bar([i + 0.2 for i in x], [stats[p]["ack_answer"] * 100 for p in precisions],
           width=0.4, label="answer span", color="#c0504d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(precisions)
    ax.set_ylabel("Acknowledgment (%)")
    ax.set_title("Thinking vs. answer acknowledgment gap")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}_gap.png", dpi=120)
    print(f"wrote {prefix}_gap.png")


def main(path):
    rows = load(path)
    prefix = path.rsplit(".", 1)[0]
    write_csv(rows, prefix + ".csv")
    precisions, stats = summarize(rows)
    make_plots(precisions, stats, prefix)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")

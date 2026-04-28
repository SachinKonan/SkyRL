"""Panel for the SFT v2 run: training loss curve + eval metrics across the
HF checkpoints (steps 50/100/150/200).
"""

import glob
import json
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

labelsize = 20
titlesize = 20
legendsize = 14
ticksize = 14

red = "#FF8988"
orange = "#FECC81"
blue = "#6098FF"
green = "#77B25D"
purple = "#B28CFF"

CKPT_ROOT = "/scratch/gpfs/ZHUANGL/sk7524/ckpts"
LOG_GLOB = "logs/sft/array_v2_7402037_*.err"
STEPS = [50, 100, 150, 200]


# ---------------------------------------------------------------------------
# (1) Loss curve from training logs
# ---------------------------------------------------------------------------
loss_re = re.compile(r"Step (\d+): loss=([\d.]+), grad_norm=([\d.]+)")
loss_by_step = {}
for path in sorted(glob.glob(LOG_GLOB)):
    with open(path) as f:
        for line in f:
            m = loss_re.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                # Latest cell wins on retrace duplicates
                loss_by_step[step] = loss

steps_sorted = sorted(loss_by_step.keys())
losses = [loss_by_step[s] for s in steps_sorted]
print(f"loss-curve steps: {len(steps_sorted)}, range [{steps_sorted[0]}, {steps_sorted[-1]}]")


# ---------------------------------------------------------------------------
# (2) Eval metrics for each v2 checkpoint
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from compare_evals import parse_row, aggregate  # type: ignore

metrics_by_step = {}
for s in STEPS:
    files = glob.glob(os.path.join(
        CKPT_ROOT,
        f"gen_eval_v2step{s}_chunk*",
        "exports", "dumped_evals", "eval_only", "iclr_arxiv*.jsonl",
    ))
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.extend(parse_row(json.loads(line)) for line in f if line.strip())
    metrics_by_step[s] = aggregate(rows) if rows else None
    print(f"step{s}: n={len(rows)}")


def m(step, key):
    return metrics_by_step[step][key] * 100  # to percent


# ---------------------------------------------------------------------------
# (3) Reference lines: full-1667-row eval for base + step206
# ---------------------------------------------------------------------------
def _full_eval(tag):
    files = glob.glob(os.path.join(
        CKPT_ROOT,
        f"gen_eval_{tag}_chunk*",
        "exports", "dumped_evals", "eval_only", "iclr_arxiv*.jsonl",
    ))
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.extend(parse_row(json.loads(line)) for line in f if line.strip())
    return aggregate(rows) if rows else None


base = _full_eval("base")
step206 = _full_eval("step206")


# ---------------------------------------------------------------------------
# Panel: 2x2
#   (0,0) loss curve
#   (0,1) accuracy + boxed_rate
#   (1,0) pred_none + parrot_rate
#   (1,1) Accept/Reject prediction balance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))


def stylize(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(r"SFT Step", fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.grid(True, linestyle="--")
    ax.tick_params(axis="both", labelsize=ticksize)
    if ylim is not None:
        ax.set_ylim(*ylim)


# --- (0,0) Loss ---
ax = axes[0, 0]
ax.plot(steps_sorted, losses, linewidth=2.5, color=blue, alpha=0.9)
stylize(ax, r"Training Loss", r"loss")
ax.set_xlim(0, max(STEPS) + 5)


# --- (0,1) Accuracy + Boxed rate ---
ax = axes[0, 1]
ax.plot(STEPS, [m(s, "accuracy") for s in STEPS],
        linewidth=3.0, marker="o", markersize=10, alpha=0.9, color=blue, label=r"accuracy")
ax.plot(STEPS, [m(s, "boxed_rate") for s in STEPS],
        linewidth=3.0, marker="s", markersize=10, alpha=0.9, color=green, label=r"boxed rate")
if base is not None:
    ax.axhline(base["accuracy"] * 100, linestyle=":", color=blue, alpha=0.5)
if step206 is not None:
    ax.axhline(step206["accuracy"] * 100, linestyle=":", color="gray", alpha=0.7)
ax.legend(fontsize=legendsize, loc="lower right")
stylize(ax, r"Accuracy and Termination", r"percent (\%)", ylim=(0, 100))


# --- (1,0) Failure-mode rates ---
ax = axes[1, 0]
ax.plot(STEPS, [m(s, "pred_none") for s in STEPS],
        linewidth=3.0, marker="v", markersize=10, alpha=0.9, color=red, label=r"no answer")
ax.plot(STEPS, [m(s, "parrot_rate") for s in STEPS],
        linewidth=3.0, marker="^", markersize=10, alpha=0.9, color=orange, label=r"parrot")
ax.legend(fontsize=legendsize, loc="upper right")
stylize(ax, r"Failure-Mode Rates", r"percent (\%)", ylim=(0, 50))


# --- (1,1) Accept / Reject balance ---
ax = axes[1, 1]
ax.plot(STEPS, [m(s, "pred_accept") for s in STEPS],
        linewidth=3.0, marker="o", markersize=10, alpha=0.9, color=green, label=r"predict Accept")
ax.plot(STEPS, [m(s, "pred_reject") for s in STEPS],
        linewidth=3.0, marker="s", markersize=10, alpha=0.9, color=purple, label=r"predict Reject")
ax.axhline(50, linestyle=":", color="black", alpha=0.5)
ax.legend(fontsize=legendsize, loc="lower right")
stylize(ax, r"Verdict Distribution", r"percent (\%)", ylim=(0, 100))


plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "v2_panel.pdf")
png_path = os.path.join(out_dir, "v2_panel.png")
plt.savefig(pdf_path, dpi=200, bbox_inches="tight", transparent=False)
plt.savefig(png_path, dpi=200, bbox_inches="tight", transparent=False)
print(f"\nwrote {pdf_path} and {png_path}")

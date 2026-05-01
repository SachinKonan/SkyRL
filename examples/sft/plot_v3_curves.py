"""Panel for the SFT v3 run: training loss + eval metrics across the
HF checkpoints (steps 50/100/150/200), one line per modality
(text, panel, vision).

Mirrors examples/sft/plot_v2_curves.py (commit a3ec13c1) but folds three
modalities onto each subplot so we can compare them side-by-side.

Gracefully skips modalities/steps whose data isn't on disk yet — e.g.
during a partial run, vision may have only training loss but no eval.

Outputs:
  examples/sft/artifacts/v3_panel.{pdf,png}
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

# LaTeX/Helvetica match v2; fall back to safer defaults if usetex isn't
# available so the script still produces a PNG on a fresh checkout.
try:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "helvetica",
    })
except Exception:
    pass

labelsize = 18
titlesize = 18
legendsize = 13
ticksize = 13

CKPT_ROOT = "/scratch/gpfs/ZHUANGL/sk7524/ckpts"
LF_PANEL_CKPT = os.path.join(CKPT_ROOT, "codex_v3_panel")
LF_VISION_CKPT = os.path.join(CKPT_ROOT, "codex_v3_vision")
TEXT_LOG_GLOB = "logs/sft/v3_text_7551285_*.err"   # skyrl loss lines

STEPS = [50, 100, 150, 200]

# --- color per modality (consistent across all 4 subplots) -------------------
COLOR = {
    "text":   "#6098FF",   # blue
    "panel":  "#FF8988",   # red
    "vision": "#77B25D",   # green
}
MARKER = {"text": "o", "panel": "s", "vision": "D"}


# ---------------------------------------------------------------------------
# (1) Training loss — per modality
# ---------------------------------------------------------------------------
def _text_loss_curve():
    """Skyrl writes 'Step N: loss=X.XXXX, grad_norm=Y' to stderr per step."""
    loss_re = re.compile(r"Step (\d+): loss=([\d.]+), grad_norm=([\d.]+)")
    by_step: dict[int, float] = {}
    for path in sorted(glob.glob(TEXT_LOG_GLOB)):
        with open(path) as f:
            for line in f:
                m = loss_re.search(line)
                if m:
                    by_step[int(m.group(1))] = float(m.group(2))   # latest cell wins
    if not by_step:
        return [], []
    steps = sorted(by_step)
    return steps, [by_step[s] for s in steps]


def _lf_loss_curve(ckpt_dir: str):
    """LLaMA-Factory writes log_history with step/loss to trainer_state.json."""
    p = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(p):
        # try to find the latest checkpoint-N/trainer_state.json
        cands = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*", "trainer_state.json")))
        if cands:
            p = cands[-1]
        else:
            return [], []
    s = json.load(open(p))
    steps, losses = [], []
    for h in s.get("log_history", []):
        if "loss" in h and "step" in h:
            steps.append(h["step"])
            losses.append(h["loss"])
    return steps, losses


# ---------------------------------------------------------------------------
# (2) Eval metrics — per modality, per step checkpoint
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from compare_evals import aggregate, parse_row  # type: ignore


def _eval_metrics(modality: str, step: int) -> dict | None:
    """Glob the per-chunk eval JSONLs that match ``v3_<modality>_step<step>``
    and aggregate. Returns None if no chunks exist yet."""
    pat = os.path.join(
        CKPT_ROOT,
        f"v3_{modality}_step{step}_chunk*",
        "exports", "dumped_evals", "eval_only", "iclr_arxiv*.jsonl",
    )
    files = glob.glob(pat)
    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(parse_row(json.loads(line)))
    if not rows:
        return None
    out = aggregate(rows)
    out["_chunks"] = len(files)
    return out


# ---------------------------------------------------------------------------
# (3) Build the panel
# ---------------------------------------------------------------------------
def stylize(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel("SFT Step", fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=ticksize)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    # Load training-loss curves
    loss_curves = {
        "text":   _text_loss_curve(),
        "panel":  _lf_loss_curve(LF_PANEL_CKPT),
        "vision": _lf_loss_curve(LF_VISION_CKPT),
    }
    for mod, (s, l) in loss_curves.items():
        n = len(s)
        rng = (s[0], s[-1]) if n else ("-", "-")
        print(f"  loss [{mod}]: {n} points, range {rng}")

    # Load eval metrics
    eval_metrics: dict[str, dict[int, dict | None]] = {m: {} for m in ("text", "panel", "vision")}
    for mod in eval_metrics:
        for step in STEPS:
            eval_metrics[mod][step] = _eval_metrics(mod, step)
        present = [s for s in STEPS if eval_metrics[mod][s] is not None]
        n_per = {s: eval_metrics[mod][s]["n"] for s in present}
        print(f"  eval [{mod}]: steps with data: {present}  n_per_step: {n_per}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # --- (0,0) Training loss --- one line per modality (right axis: text shares y axis)
    ax = axes[0, 0]
    text_steps, text_losses = loss_curves["text"]
    panel_steps, panel_losses = loss_curves["panel"]
    vision_steps, vision_losses = loss_curves["vision"]

    # Text loss is on a different scale (~0.2) than VL (~2). Use a twin-y.
    ax2 = ax.twinx()
    if text_steps:
        ax.plot(text_steps, text_losses, linewidth=2.0, alpha=0.9,
                color=COLOR["text"], label="text (left axis)")
    if panel_steps:
        ax2.plot(panel_steps, panel_losses, linewidth=2.0, alpha=0.9,
                 color=COLOR["panel"], label="panel (right axis)")
    if vision_steps:
        ax2.plot(vision_steps, vision_losses, linewidth=2.0, alpha=0.9,
                 color=COLOR["vision"], label="vision (right axis)")
    ax.set_xlabel("SFT Step", fontsize=labelsize)
    ax.set_ylabel("text loss", fontsize=labelsize, color=COLOR["text"])
    ax2.set_ylabel("VL loss (panel + vision)", fontsize=labelsize)
    ax.set_title("Training Loss", fontsize=titlesize)
    ax.tick_params(axis="y", labelcolor=COLOR["text"], labelsize=ticksize)
    ax2.tick_params(axis="y", labelsize=ticksize)
    ax.tick_params(axis="x", labelsize=ticksize)
    ax.grid(True, linestyle="--", alpha=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=legendsize, loc="upper right")
    if any([text_steps, panel_steps, vision_steps]):
        all_max = max(([max(text_steps)] if text_steps else [])
                      + ([max(panel_steps)] if panel_steps else [])
                      + ([max(vision_steps)] if vision_steps else []))
        ax.set_xlim(0, all_max + 5)

    # Helper: plot one metric across modalities with available data
    def plot_metric(ax, key, label, ylim=None, scale=100, show_legend=True):
        for mod in ("text", "panel", "vision"):
            xs, ys = [], []
            for s in STEPS:
                m = eval_metrics[mod][s]
                if m is not None and key in m:
                    xs.append(s)
                    ys.append(m[key] * scale)
            if xs:
                ax.plot(xs, ys, linewidth=2.5, marker=MARKER[mod], markersize=9,
                        alpha=0.9, color=COLOR[mod], label=mod)
        stylize(ax, label["title"], label["y"], ylim=ylim)
        if show_legend:
            ax.legend(fontsize=legendsize, loc="lower right")

    # --- (0,1) Accuracy ---
    plot_metric(
        axes[0, 1], "accuracy",
        {"title": "Accuracy", "y": "percent (%)"},
        ylim=(0, 100),
    )
    # Add chance line at 50% (binary task; balanced eval set)
    axes[0, 1].axhline(50, linestyle=":", color="black", alpha=0.4)

    # --- (1,0) Format / no-answer rate (boxed_rate) ---
    plot_metric(
        axes[1, 0], "boxed_rate",
        {"title": "Boxed-Verdict Rate (format compliance)", "y": "percent (%)"},
        ylim=(0, 100),
    )

    # --- (1,1) Predict-Accept fraction ---
    plot_metric(
        axes[1, 1], "pred_accept",
        {"title": "Verdict Distribution: predict Accept", "y": "percent (%)"},
        ylim=(0, 100),
    )
    axes[1, 1].axhline(50, linestyle=":", color="black", alpha=0.4)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "v3_panel.pdf")
    png_path = os.path.join(out_dir, "v3_panel.png")
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight", transparent=False)
    plt.savefig(png_path, dpi=200, bbox_inches="tight", transparent=False)
    print(f"\nwrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()

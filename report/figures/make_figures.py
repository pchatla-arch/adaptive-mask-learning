"""Generate the schematic figures and the Table-3 chart for the report.

Only figures that contain no experimental data (diagrams) or that are a direct
visualization of numbers already in the report (Table 3) are produced here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

INK = "#222222"
BLUE = "#3b5b92"
GREEN = "#2f7d5c"
ORANGE = "#c2652a"
GREY = "#888888"
LIGHT = "#f2f2f2"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9.5, bold=False, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK, style="-|>", lw=1.3, ls="-", rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw,
                        mutation_scale=12, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


# --------------------------------------------------------------------------
# Figure 1: pipeline (three stages, left to right)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14.5, 4.1))
ax.set_xlim(0, 15.0)
ax.set_ylim(0, 4.0)
ax.axis("off")

PW = 4.2           # panel width
GAP = 0.9         # gap between panels
X = [0.25, 0.25 + PW + GAP, 0.25 + 2 * (PW + GAP)]
ROW = [2.45, 1.45, 0.45]   # y of top / middle / bottom boxes
BH = 0.6

for x0, title, col in [(X[0], r"Stage 1 · Tracker $\pi_{FC}$", BLUE),
                       (X[1], r"Stage 2 · Inpainter $\pi_{PC}$", GREEN),
                       (X[2], r"Stage 3 · Mask scheduler $\pi_\theta$", ORANGE)]:
    ax.add_patch(FancyBboxPatch((x0, 0.25), PW, 3.45, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=LIGHT, ec=col, lw=1.6))
    ax.text(x0 + PW / 2, 3.42, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=col)

def col_boxes(x0, texts, ec_list=None):
    ec_list = ec_list or [INK] * 3
    for y, t, ec in zip(ROW, texts, ec_list):
        box(ax, x0 + 0.22, y, PW - 0.44, BH, t, fs=8.6, ec=ec)

# Stage 1 (top→bottom flow)
col_boxes(X[0], ["AMASS full-body reference motion",
                 r"$\pi_{FC}$ — PPO, IsaacGym" + "\ninit. from ProtoMotions checkpoint",
                 r"full-body goals $g_t^{full}$ + expert actions"])
cx = X[0] + PW / 2
arrow(ax, cx, ROW[0], cx, ROW[1] + BH + 0.02)
arrow(ax, cx, ROW[1], cx, ROW[2] + BH + 0.02)

# Stage 2 (top→bottom flow)
col_boxes(X[1], [r"mask  $M_{\rho_t}(g_t^{full}) \rightarrow g_t^{partial}$",
                 r"$\pi_{PC}$ — transformer (6 layers, 8 heads)" + "\nDAgger step on masked batch",
                 "predicted poses → physics sim\n→ validation every 10k steps"])
cx = X[1] + PW / 2
arrow(ax, cx, ROW[0], cx, ROW[1] + BH + 0.02)
arrow(ax, cx, ROW[1], cx, ROW[2] + BH + 0.02)

# Stage 3: top = policy, middle = its inputs, bottom = reward
col_boxes(X[2], [r"$\pi_\theta$: 2-layer MLP (128) → Beta($\alpha,\beta$) → $\rho_t$",
                 "training-state input\n" + r"$\Delta$ val loss · grad norm · mask entropy",
                 r"$R_t = 1.0\,\Delta\mathrm{ValLoss} + 0.5\,\mathrm{acc}$  (EMA)"],
          ec_list=[ORANGE, INK, INK])
cx = X[2] + PW / 2
arrow(ax, cx, ROW[1] + BH, cx, ROW[0] - 0.02, color=ORANGE)          # inputs → policy
# reward → PPO update → policy (route along the right edge)
rx = X[2] + PW - 0.10
arrow(ax, X[2] + PW - 0.22, ROW[2] + BH / 2, rx, ROW[2] + BH / 2, color=ORANGE, style="-")
arrow(ax, rx, ROW[2] + BH / 2, rx, ROW[0] + BH / 2, color=ORANGE, style="-")
arrow(ax, rx, ROW[0] + BH / 2, X[2] + PW - 0.22, ROW[0] + BH / 2, color=ORANGE)
ax.text(rx + 0.16, ROW[1] + BH / 2, "PPO update", fontsize=7.5, color=ORANGE, ha="center", va="center", rotation=90)

# Cross-stage arrows (all horizontal, no crossings)
def hlabel(x, y, s, color=INK, dy=0.13):
    ax.text(x, y + dy, s, fontsize=8, ha="center", va="bottom", color=color)

# Stage1 bottom → Stage2 top: goals feed the mask
arrow(ax, X[0] + PW - 0.22, ROW[2] + BH / 2, X[1] + 0.22, ROW[2] + BH / 2, color=GREY, ls="--")
hlabel((X[0] + PW + X[1]) / 2, ROW[2] + BH / 2, r"$g_t^{full}$, expert labels", color=GREY, dy=0.08)
# Stage3 top → Stage2 top: rho sets the mask budget
arrow(ax, X[2] + 0.22, ROW[0] + BH / 2, X[1] + PW - 0.22, ROW[0] + BH / 2, color=ORANGE)
hlabel((X[1] + PW + X[2]) / 2, ROW[0] + BH / 2, r"$\rho_t \in [0.1,\,0.9]$", color=ORANGE, dy=0.08)
# Stage2 middle → Stage3 middle: training signals
arrow(ax, X[1] + PW - 0.22, ROW[1] + BH / 2, X[2] + 0.22, ROW[1] + BH / 2, color=INK)
hlabel((X[1] + PW + X[2]) / 2, ROW[1] + BH / 2, "training signals", dy=0.08)
# Stage2 bottom → Stage3 bottom: validation metrics
arrow(ax, X[1] + PW - 0.22, ROW[2] + BH / 2, X[2] + 0.22, ROW[2] + BH / 2, color=INK)
hlabel((X[1] + PW + X[2]) / 2, ROW[2] + BH / 2, "val. metrics", dy=0.08)

fig.savefig("fig1_pipeline.png", dpi=220, bbox_inches="tight")
fig.savefig("fig1_pipeline.pdf", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: scheduler architecture and training loop
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.4)
ax.axis("off")

# Left: policy network
ax.text(1.9, 4.1, "Masking policy $\\pi_\\theta$", ha="center", fontsize=10.5, fontweight="bold", color=ORANGE)
inputs = [r"$\Delta$ validation loss", "mask entropy", "gradient norm"]
for i, name in enumerate(inputs):
    y = 3.15 - i * 0.62
    box(ax, 0.3, y, 1.55, 0.45, name, fs=8.8)
    arrow(ax, 1.85, y + 0.225, 2.45, 2.55 + (1 - i) * 0.0 + 0.0 * i, color=GREY, lw=1.0)
box(ax, 2.45, 2.15, 1.35, 0.85, "MLP\n2 layers × 128", fc="#fbe9dc", ec=ORANGE, fs=8.8)
arrow(ax, 3.8, 2.575, 4.35, 2.575)
box(ax, 4.35, 2.15, 1.35, 0.85, r"Beta($\alpha, \beta$)" + "\nsample", fc="#fbe9dc", ec=ORANGE, fs=8.8)
arrow(ax, 5.7, 2.575, 6.25, 2.575)
box(ax, 6.25, 2.15, 1.5, 0.85, "affine map\n" + r"$\rho_t \in [0.1,\,0.9]$", fc="#fbe9dc", ec=ORANGE, fs=8.8)

# Right: inpainter it drives
box(ax, 8.3, 2.15, 2.4, 0.85, "mask generator $M_{\\rho_t}$\n(structured sampler,\nbudget set by $\\rho_t$)", fs=8.6, ec=GREEN)
arrow(ax, 7.75, 2.575, 8.3, 2.575, color=ORANGE)
box(ax, 8.3, 0.85, 2.4, 0.85, r"$\pi_{PC}$ transformer" + "\n6 layers · 8 heads\nDAgger step on masked batch", fs=8.6, ec=GREEN)
arrow(ax, 9.5, 2.15, 9.5, 1.7, color=GREEN)

# Bottom: reward / PPO loop
box(ax, 4.35, 0.85, 3.4, 0.85,
    "validation every 10k steps\n" + r"$R_t = \alpha(\mathrm{ValLoss}_{t-1}-\mathrm{ValLoss}_t) + \beta\,\mathrm{acc}_t$" + "\n" + r"$\alpha = 1.0,\ \beta = 0.5$, EMA-smoothed",
    fs=8.4, ec=BLUE)
arrow(ax, 8.3, 1.275, 7.75, 1.275, color=BLUE)
arrow(ax, 4.35, 1.275, 3.1, 1.275, color=BLUE)
arrow(ax, 3.1, 1.275, 3.1, 2.15, color=BLUE)
ax.text(2.55, 1.1, "PPO update\nof $\\theta$", fontsize=8.3, color=BLUE, ha="center", va="top")
arrow(ax, 6.05, 0.85, 6.05, 0.35, color=GREY, lw=1.0)
arrow(ax, 6.05, 0.35, 1.05, 0.35, color=GREY, lw=1.0, style="-")
arrow(ax, 1.05, 0.35, 1.05, 1.9, color=GREY, lw=1.0)
ax.text(3.5, 0.2, "next training-state observation", fontsize=8, color=GREY, ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=1))

ax.text(9.5, 4.1, "Inpainting controller (Stage 2)", ha="center", fontsize=10.5, fontweight="bold", color=GREEN)
ax.text(9.5, 3.55, "trained under the scheduled mask", ha="center", fontsize=8.5, color=GREY)

fig.savefig("fig2_architecture.png", dpi=220, bbox_inches="tight")
fig.savefig("fig2_architecture.pdf", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# Figure: Table 3 as small multiples
# --------------------------------------------------------------------------
methods = ["Random", "Cosine", "Adaptive\n(ours)"]
colors = [GREY, BLUE, ORANGE]
metrics = [
    ("L1 loss ↓", [0.36, 0.31, 0.10], "{:.2f}"),
    ("MPJPE (mm) ↓", [47.2, 42.7, 29.6], "{:.1f}"),
    ("FID ↓", [29.5, 23.1, 12.4], "{:.1f}"),
    ("Action accuracy (%) ↑", [76.4, 80.5, 92.1], "{:.1f}"),
    ("Foot-slip error (%) ↓", [12.8, 10.3, 3.2], "{:.1f}"),
]
fig, axes = plt.subplots(1, 5, figsize=(12, 2.9))
for ax, (title, vals, fmt) in zip(axes, metrics):
    bars = ax.bar(methods, vals, color=colors, width=0.62, edgecolor="none")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02, fmt.format(v),
                ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#e5e5e5", lw=0.7)
    ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.2)
fig.savefig("fig_metrics_comparison.png", dpi=220, bbox_inches="tight")
fig.savefig("fig_metrics_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print("done")

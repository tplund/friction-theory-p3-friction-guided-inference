"""Generate all figures for Paper 3.

Produces PNG files in docs/figures/ for inclusion in paper.

Figures:
1. Main results table (bar chart with CI)
2. Oracle depth plateau curves (all cells)
3. Strategy effectiveness heatmap
4. Abstention curve (accuracy vs coverage)
5. Combined strategy+abstention comparison
"""
import io
import json
import os
import re
import sys
import random
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

LETTER_RE = re.compile(r'\b([A-J])\b')


# ============================================================
# Figure 1: Main results bar chart
# ============================================================
def fig1_main_results():
    cells = [
        ("Qwen2.5-7B\nMATH-500", 45.8, 66.5, 15.1, 26.4),
        ("Qwen3-235B\nSimpleQA", 41.1, 51.8, 9.1, 12.1),
        ("Qwen3-235B\nMMLU-Pro", 55.2, 62.9, 6.4, 9.0),
        ("LiquidAI\nMMLU-Pro", 33.8, 41.9, 7.1, 9.2),
        ("GPT-oss-20B\nGPQA*", 27.0, 30.4, 0.0, 6.8),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(cells))
    w = 0.35

    vanilla_bars = [c[1] for c in cells]
    adaptive_bars = [c[2] for c in cells]
    ci_lo = [c[2] - c[1] - c[3] for c in cells]  # lift - ci_low
    ci_hi = [c[4] - (c[2] - c[1]) for c in cells]  # ci_high - lift
    lifts = [c[2] - c[1] for c in cells]

    bars1 = ax.bar([i - w/2 for i in x], vanilla_bars, w, label="Vanilla", color="#4A90D9", alpha=0.8)
    bars2 = ax.bar([i + w/2 for i in x], adaptive_bars, w, label="Adaptive", color="#E74C3C", alpha=0.8)

    # Add lift annotations
    for i, (name, van, adp, ci_l, ci_h) in enumerate(cells):
        lift = adp - van
        ax.annotate(f"+{lift:.1f}pp", xy=(i + w/2, adp + 1), ha="center", fontsize=9, fontweight="bold", color="#E74C3C")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Friction-Guided Inference: 5 Cells, 4 Architectures, All Significant")
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cells], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_main_results.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Figure 2: Depth plateau curves
# ============================================================
def fig2_depth_plateaus():
    depth_data = {
        "SQA Qwen3\n(22 rounds)": [34, 56, 56, 56, 56, 58, 58, 58, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
        "MMLU-Pro Qwen3\n(SBS x20)": [54, 90, 90, 90, 90],
        "LiquidAI\n(verify x20)": [30, 64, 72, 74, 74, 76, 76, 76, 78, 80, 80, 82, 84, 84, 84, 84, 84, 84, 86, 86, 86],
        "GPT-oss GPQA\n(SBS x20)": [26, 38, 44, 50, 50, 50, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 54, 54],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C", "#4A90D9", "#2ECC71", "#F39C12"]

    for (label, data), color in zip(depth_data.items(), colors):
        rounds = list(range(1, len(data) + 1))
        ax.plot(rounds, data, marker="o", markersize=3, color=color, label=label, linewidth=2)

    ax.set_xlabel("Round")
    ax.set_ylabel("Oracle accuracy (%)")
    ax.set_title("Depth Calibration: Oracle Plateau per Cell")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 22)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_depth_plateaus.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Figure 3: Strategy effectiveness heatmap
# ============================================================
def fig3_strategy_heatmap():
    # Net rescues per strategy per cell (from calibration data)
    strategies = ["step_by_step", "verify", "narrow", "challenge", "pre_mortem"]
    cells = ["Qwen2.5\nMATH", "Qwen3\nSQA", "Qwen3\nMMLU-Pro", "LiquidAI\nMMLU-Pro", "GPT-oss\nGPQA"]

    # Net values (positive = constructive, negative = destructive)
    data = [
        # sbs  verify  narrow  challenge  pre_mortem
        [+11,    0,     -5,      0,         0],     # MATH qwen2.5
        [+29,   -2,     -6,    -13,        -6],     # SQA Qwen3
        [+4,    -4,     +3,    -13,        +3],     # MMLU-Pro Qwen3
        [+2,    +6,     +3,     +0,        -2],     # MMLU-Pro LiquidAI
        [+2,    +2,     -2,     -2,        -8],     # GPQA GPT-oss
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color: green for positive, red for negative, white for zero
    import numpy as np
    arr = np.array(data)
    max_abs = max(abs(arr.min()), abs(arr.max()))

    im = ax.imshow(arr.T, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=10)

    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(cells)):
            val = data[j][i]
            color = "white" if abs(val) > max_abs * 0.6 else "black"
            text = f"+{val}" if val > 0 else str(val)
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_title("Strategy Effectiveness (net rescues - regressions)")
    fig.colorbar(im, ax=ax, label="Net (green=constructive, red=destructive)")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_strategy_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Figure 4: Abstention curve
# ============================================================
def fig4_abstention():
    # Pre-computed from analysis
    abstention_data = {
        "MATH Qwen2.5-7B": [(0, 46.9), (10, 53.4), (20, 60.3), (30, 66.4)],
        "SQA Qwen3-235B": [(0, 41.6), (10, 49.4), (20, 55.5), (30, 61.8)],
        "MMLU-Pro Qwen3": [(0, 55.2), (10, 60.8), (20, 61.7), (30, 61.4)],
        "MMLU-Pro LiquidAI": [(0, 33.8), (10, 40.1), (20, 45.4), (30, 50.3)],
        "GPQA GPT-oss": [(0, 26.3), (10, 31.3), (20, 40.4), (30, 48.5)],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E74C3C", "#4A90D9", "#2ECC71", "#F39C12", "#9B59B6"]

    for (label, data), color in zip(abstention_data.items(), colors):
        pcts = [d[0] for d in data]
        accs = [d[1] for d in data]
        ax.plot(pcts, accs, marker="o", color=color, label=label, linewidth=2)

    ax.set_xlabel("Abstention rate (%)")
    ax.set_ylabel("Success rate (% correct + correctly abstained)")
    ax.set_title("CR-Based Abstention: Success Rate vs Coverage")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, 32)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_abstention.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Figure 5: Combined strategy + abstention comparison
# ============================================================
def fig5_combined():
    cells = ["SQA\nQwen3", "MMLU-Pro\nQwen3", "MMLU-Pro\nLiquidAI", "GPQA\nGPT-oss"]
    vanilla = [41.6, 55.2, 33.8, 26.3]
    strategy_only = [52.5, 64.8, 48.1, 31.3]
    abstention_only = [55.5, 61.7, 45.4, 40.4]
    combined = [57.2, 67.4, 55.0, 45.5]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(cells))
    w = 0.2

    ax.bar([i - 1.5*w for i in x], vanilla, w, label="Vanilla", color="#95A5A6", alpha=0.8)
    ax.bar([i - 0.5*w for i in x], strategy_only, w, label="Strategy only", color="#4A90D9", alpha=0.8)
    ax.bar([i + 0.5*w for i in x], abstention_only, w, label="Abstention only", color="#F39C12", alpha=0.8)
    ax.bar([i + 1.5*w for i in x], combined, w, label="Strategy + Abstention", color="#E74C3C", alpha=0.8)

    # Add combined lift annotations
    for i in range(len(cells)):
        lift = combined[i] - vanilla[i]
        ax.annotate(f"+{lift:.1f}pp", xy=(i + 1.5*w, combined[i] + 1), ha="center", fontsize=8, fontweight="bold", color="#E74C3C")

    ax.set_ylabel("Success rate (%)")
    ax.set_title("Combined Strategy + Abstention: Best of Both Worlds")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 80)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_combined.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Figure 6: Semantics factorial decomposition (2-panel)
# ============================================================
def fig6_semantics_factorial():
    # Qwen3 x MMLU-Pro (n=50)
    qwen3 = [
        ("D1 direct-modal-defend",   68.0, 54.0, 80.0, "defend"),
        ("I5 imag-modal-defend",     68.0, 54.0, 80.0, "defend"),
        ("I7 imag-decl-defend",      64.0, 50.0, 78.0, "defend"),
        ("D3 direct-decl-defend",    62.0, 48.0, 74.0, "defend"),
        ("D4 direct-decl-reframe",   56.0, 42.0, 70.0, "reframe"),
        ("C0 neutral",               54.0, 40.0, 68.0, "neutral"),
        ("T4 one-year",              54.0, 40.0, 68.0, "reframe"),
        ("D2 direct-modal-reframe",  48.0, 34.0, 62.0, "reframe"),
        ("T2 tomorrow",              46.0, 32.0, 60.0, "reframe"),
        ("T1 no-temporal",           44.0, 28.0, 58.0, "reframe"),
        ("I8 imag-decl-reframe*",    42.0, 28.0, 56.0, "reframe"),
        ("T3 yesterday",             30.0, 16.0, 44.0, "reframe"),
        ("I6 imag-modal-reframe",    30.0, 16.0, 44.0, "reframe"),
    ]
    # Cogito x GPQA Diamond (n=50)
    cogito = [
        ("D3 direct-decl-defend",    28.0, 14.0, 42.0, "defend"),
        ("I8 imag-decl-reframe*",    20.0,  4.0, 36.0, "reframe"),
        ("T1 no-temporal",           20.0,  4.0, 36.0, "reframe"),
        ("T4 one-year",              20.0,  2.0, 38.0, "reframe"),
        ("I5 imag-modal-defend",     16.0,  2.0, 30.0, "defend"),
        ("I7 imag-decl-defend",      16.0,  4.0, 28.0, "defend"),
        ("D1 direct-modal-defend",   12.0, -2.0, 26.0, "defend"),
        ("I6 imag-modal-reframe",    12.0, -4.0, 28.0, "reframe"),
        ("D2 direct-modal-reframe",  10.0, -4.0, 24.0, "reframe"),
        ("C0 neutral",               10.0, -4.0, 24.0, "neutral"),
        ("T2 tomorrow",               8.0, -6.0, 22.0, "reframe"),
        ("D4 direct-decl-reframe",    6.0, -10.0, 22.0, "reframe"),
        ("T3 yesterday",              4.0, -10.0, 18.0, "reframe"),
    ]

    colours = {"defend": "#2ECC71", "reframe": "#E74C3C", "neutral": "#95A5A6"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, data, title in [
        (axes[0], qwen3, "Qwen3-235B × MMLU-Pro (n=50)"),
        (axes[1], cogito, "Cogito-671B × GPQA Diamond (n=50)"),
    ]:
        y_pos = list(range(len(data)))
        y_pos.reverse()
        for i, (name, nr, lo, hi, task) in zip(y_pos, data):
            ax.barh(i, nr, xerr=[[nr - lo], [hi - nr]], color=colours[task],
                    alpha=0.85, capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([v[0] for v in data], fontsize=8)
        ax.set_xlabel("Net rescue rate (pp), 95% CI")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0, color="black", linewidth=0.5)

    # Shared legend
    handles = [
        mpatches.Patch(color=colours["defend"], label="Task: defend"),
        mpatches.Patch(color=colours["reframe"], label="Task: reframe"),
        mpatches.Patch(color=colours["neutral"], label="Neutral control"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Semantic decomposition of pre-mortem — component-level rescue ranking",
                 fontsize=12, y=1.00)
    fig.text(0.5, 0.02, "* classical pre-mortem (imagined + declarative + reframe)",
             fontsize=8, ha="center", style="italic", color="#555")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_semantics_factorial.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("Generating Paper 3 figures...")
    fig1_main_results()
    fig2_depth_plateaus()
    fig3_strategy_heatmap()
    fig4_abstention()
    fig5_combined()
    fig6_semantics_factorial()
    print(f"\nAll figures saved to {FIG_DIR}/")

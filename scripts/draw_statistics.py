import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Raw fold values
# -----------------------------
f1_baseline = np.array([0.6955, 0.7355, 0.6943, 0.7599, 0.7114])
f1_cbam     = np.array([0.7455, 0.7376, 0.7137, 0.7632, 0.7195])

rec_baseline = np.array([0.6346, 0.6593, 0.6198, 0.6840, 0.6420])
rec_cbam     = np.array([0.6617, 0.7185, 0.6370, 0.7160, 0.6840])

# -----------------------------
# Plot style
# -----------------------------
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14
})

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Professional color palette
colors = ["#2E5EAA", "#D1495B"]   # deep blue & muted red
scatter_colors = ["#1B3B6F", "#9C1C2B"]

def draw_panel(ax, baseline, cbam, ylabel):
    means = [baseline.mean(), cbam.mean()]
    stds  = [baseline.std(ddof=1), cbam.std(ddof=1)]
    x = np.array([0, 1])

    # Bars
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        width=0.55,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
        zorder=2
    )

    # Paired lines + scatter
    for i in range(len(baseline)):
        ax.plot(
            x,
            [baseline[i], cbam[i]],
            color="gray",
            linewidth=1,
            alpha=0.6,
            zorder=1
        )
        ax.scatter(
            x,
            [baseline[i], cbam[i]],
            color=scatter_colors,
            edgecolor="black",
            s=50,
            zorder=3
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "ResNet18 + CBAM"])
    ax.set_ylabel(ylabel)

    ymin = min(baseline.min(), cbam.min()) - 0.02
    ymax = max(baseline.max(), cbam.max()) + 0.02
    ax.set_ylim(ymin, ymax)

# Draw both panels
draw_panel(axes[0], f1_baseline, f1_cbam, "F1 Score")
draw_panel(axes[1], rec_baseline, rec_cbam, "Recall")

axes[0].set_title("F1 Score Comparison")
axes[1].set_title("Recall Comparison")

plt.tight_layout()

# High-resolution output
plt.savefig("cbam_vs_baseline_f1_recall.png", dpi=600, bbox_inches="tight")
plt.savefig("cbam_vs_baseline_f1_recall.pdf", dpi=600, bbox_inches="tight")

plt.show()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

epochs = [0, 10, 20, 30, 39]
loss = {
    0: [4.3459, 3.2281, 3.0398, 2.8311, 2.7853],
    1: [4.3145, 3.3338, 3.0454, 3.0291, 2.8751],
    2: [4.3516, 3.0969, 2.9864, 2.7873, 2.6999],
    3: [4.3539, 3.1558, 3.0164, 2.8884, 12.5702],
    4: [4.3547, 3.1027, 2.9712, 2.8875, 2.9158],
}

GRAY, ACCENT, INK, MUTED = "#9AA0A6", "#D95F02", "#333333", "#707070"

fig, ax = plt.subplots(figsize=(5.4, 3.1))
for s in [0, 1, 2, 4]:
    ax.plot(epochs, loss[s], color=GRAY, lw=1.6, marker="o", ms=4,
            solid_capstyle="round", zorder=2)
ax.plot(epochs, loss[3], color=ACCENT, lw=1.8, marker="o", ms=4.5,
        solid_capstyle="round", zorder=3)

ax.annotate("seed 3", xy=(39, 12.5702), xytext=(34.0, 12.55),
            color=ACCENT, fontsize=9.5, ha="right", va="center")
ax.annotate("seeds 0, 1, 2, 4", xy=(39, 2.83), xytext=(38.6, 2.42),
            color=MUTED, fontsize=9.5, ha="right", va="top")

ax.set_yscale("log")
ax.set_yticks([3, 4, 6, 12])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(2.2, 15)
ax.set_xticks(epochs)
ax.set_xlim(-1.5, 40.5)
ax.set_xlabel("epoch")
ax.set_ylabel("training loss (per-epoch mean)")
ax.grid(axis="y", color="#E3E3E3", lw=0.7, zorder=0)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color("#B0B0B0")
ax.tick_params(colors=MUTED)
ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)

fig.tight_layout()
fig.savefig("/home/yipjiaqi/world-encoder/paper/figures/training_loss.pdf")
fig.savefig("/tmp/training_loss_preview.png", dpi=160)
print("saved")

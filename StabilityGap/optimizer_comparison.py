import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

optimizers = {
    "AdaGrad": "perf_adagrad.csv",
    "RMSprop": "perf_rmsprop.csv",
    "Adam": "perf_adam.csv"
}

CONFIG = {
    "x_range_zoom": (490, 650),
    "y_range_zoom": (90, 97.5),
    "n_tasks": 4,
    "iters_per_task": 500,
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig = plt.figure(figsize=(12, 8))
ax = fig.add_axes([0.1, 0.08, 0.85, 0.75])
axins1 = fig.add_axes([0.25, 0.85, 0.3, 0.116])

for (label, file), color in zip(optimizers.items(), colors):
    df = pd.read_csv(f"./store/data/{file}")
    ax.plot(df["iteration"], df["mean_perf"], label=label, color=color, linewidth=1.5)
    ax.fill_between(
        df["iteration"],
        df["mean_perf"] - df["stderr_perf"],
        df["mean_perf"] + df["stderr_perf"],
        color=color,
        alpha=0.3
    )

ax.set_xticks(range(0, 2001, 500))
ax.set_yticks(range(80, 101, 5))

ax.set_xlabel("Iterations", fontsize=20)
ax.set_ylabel("Test Accuracy on Task 1 (%)", fontsize=20)
ax.set_ylim(80, 100)
ax.set_xlim(0, 2000)
ax.legend(loc='lower right', frameon=True, facecolor='#f2f2f2', edgecolor='#e0e0e0', fontsize=15, framealpha=1, borderpad=1)

n_tasks = 4
iters = 500
for switch_id in range(1, n_tasks):
    ax.axvline(x=iters * switch_id, color='gray', linestyle='--', label='Task switch' if switch_id == 1 else "")

for (label, file), color in zip(optimizers.items(), colors):
    df = pd.read_csv(f"./store/data/{file}")

    smooth_mean = df["mean_perf"]
    smooth_err = df["stderr_perf"]

    axins1.plot(df["iteration"], smooth_mean, color=color)
    axins1.fill_between(
        df["iteration"],
        smooth_mean - smooth_err,
        smooth_mean + smooth_err,
        color=color,
        alpha=0.3
    )

x1, x2 = CONFIG['x_range_zoom']
y1, y2 = CONFIG['y_range_zoom']

axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)

axins1.set_autoscale_on(False)

axins1.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True, labelsize=12)
axins1.tick_params(axis='y', labelsize=12)

ax.tick_params(axis='x', labelsize=19.5)
ax.tick_params(axis='y', labelsize=19.5)

axins1.set_yticks([y1, y2])
axins1.set_xticks([x1, x2])

axins1.set_facecolor("#f2f2f2")

rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=1, edgecolor='black',
                         facecolor='none', linestyle='--')
ax.add_patch(rect)

con1 = ConnectionPatch(xyA=(x1, y1), xyB=(x1, y2),
                       coordsA="data", coordsB="data",
                       axesA=axins1, axesB=ax,
                       color="black", linewidth=1, linestyle='--')
con2 = ConnectionPatch(xyA=(x2, y1), xyB=(x2, y2),
                       coordsA="data", coordsB="data",
                       axesA=axins1, axesB=ax,
                       color="black", linewidth=1, linestyle='--')

fig.add_artist(con1)
fig.add_artist(con2)

fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9)

fig.tight_layout()
fig.savefig("./store/data/optimizer_comparison.pdf")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

df = pd.read_csv("scaling_data.csv")

strong = df[df["scaling_type"] == "strong"].sort_values("tasks")
weak = df[df["scaling_type"] == "weak"].sort_values("tasks")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

def format_axes(ax, tasks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(tasks)
    ax.set_xticklabels(tasks)

    ax.set_yscale("log")

    y_ticks = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 16]  # add more as needed
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.2)

# Strong scaling
axes[0].plot(strong["tasks"], strong["runtime"], marker="o", linewidth=2, label="Runtime")
axes[0].plot(strong["tasks"], strong["compute_time"], marker="s", linewidth=2, label="Compute time")
axes[0].plot(strong["tasks"], strong["comm_time"], marker="^", linewidth=2, label="Comm time")
axes[0].set_title("Strong scaling")
axes[0].set_xlabel("Tasks")
axes[0].set_ylabel("Time (s)")
format_axes(axes[0], strong["tasks"].unique())
axes[0].legend()

# Weak scaling
axes[1].plot(weak["tasks"], weak["runtime"], marker="o", linewidth=2, label="Runtime")
axes[1].plot(weak["tasks"], weak["compute_time"], marker="s", linewidth=2, label="Compute time")
axes[1].plot(weak["tasks"], weak["comm_time"], marker="^", linewidth=2, label="Comm time")
axes[1].set_title("Weak scaling")
axes[1].set_xlabel("Tasks")
axes[1].set_ylabel("Time (s)")
format_axes(axes[1], weak["tasks"].unique())
axes[1].legend()

plt.show()
plt.savefig("benchmarks.png", dpi=600)

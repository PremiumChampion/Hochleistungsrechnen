import matplotlib.pyplot as plt

# Data
ranks = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125]
runtime_s = [43.0904, 21.5256, 11.5014, 9.80591, 6.35969, 6.21572, 7.17903, 7.71452, 11.6942, 13.8975, 18.9978, 19.1994]
performance_mlups = [23.207, 46.4564, 86.9456, 101.979, 157.24, 160.882, 139.295, 129.626, 85.5128, 71.9555, 52.6377, 52.085]

fig, ax1 = plt.subplots(figsize=(11, 6))

# Left axis: runtime
color1 = "tab:blue"
ax1.set_xlabel("Number of ranks", fontsize=12)
ax1.set_ylabel("Runtime (s)", color=color1, fontsize=12)
l1 = ax1.plot(ranks, runtime_s, marker="o", linewidth=2.5, color=color1, label="Runtime (s)")
ax1.tick_params(axis="y", labelcolor=color1, labelsize=11)
ax1.tick_params(axis="x", labelsize=11)
ax1.grid(True, linestyle="--", alpha=0.35)

# Right axis: performance
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Performance (MLUPS)", color=color2, fontsize=12)
l2 = ax2.plot(ranks, performance_mlups, marker="s", linewidth=2.5, color=color2, label="Performance (MLUPS)")
ax2.tick_params(axis="y", labelcolor=color2, labelsize=11)

# Title
plt.title(
    "Fluid Simulation Scaling: 1000×1000 Matrix, 1000 Steps (BW)\n"
    "Runtime and Performance vs. Number of Ranks",
    fontsize=14
)

# Combine legends
lines = l1 + l2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="best", fontsize=11)

plt.tight_layout()
plt.savefig("simulation_scaling.png", dpi=300, bbox_inches="tight")
plt.show()

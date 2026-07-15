import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# Load data
df = pd.read_csv("scaling_data_gpu_a100_short.csv")

# Filter into 4 separate dataframes
strong_1d = df[(df["scaling_type"] == "strong") & (df["dim"] == 1)].sort_values("tasks")
weak_1d   = df[(df["scaling_type"] == "weak")   & (df["dim"] == 1)].sort_values("tasks")
strong_2d = df[(df["scaling_type"] == "strong") & (df["dim"] == 2)].sort_values("tasks")
weak_2d   = df[(df["scaling_type"] == "weak")   & (df["dim"] == 2)].sort_values("tasks")

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Main title
fig.suptitle("Fluid Simulation Scaling Benchmarks: bwUniCluster (gpu_a100_short)", 
             fontsize=16, fontweight='bold')

def format_axes(ax, tasks):
    if len(tasks) == 0:
        return
    
    ax.set_xscale("log", base=2)
    ax.set_xticks(tasks)
    ax.set_xticklabels(tasks)

    ax.set_yscale("log")
    y_ticks = [0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 16, 32]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle="--", alpha=0.2)

def plot_benchmark(ax, df_subset, scaling_type, dim_str):
    """Helper function to plot a single subplot safely."""
    
    # Handle empty datasets gracefully if some jobs haven't finished
    if df_subset.empty:
        ax.set_title(f"{scaling_type} Scaling ({dim_str})\n(No data available)", fontsize=12)
        ax.text(0.5, 0.5, 'Incomplete/No Data', horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes, color='gray')
        ax.set_xlabel("Tasks [count]")
        ax.set_ylabel("Time [s]")
        return

    # Extract problem sizes dynamically from the CSV data
    if scaling_type == "Strong":
        nx = int(df_subset["Nx"].iloc[0])
        ny = int(df_subset["Ny"].iloc[0])
        subtitle = f"(Global Grid: {nx} × {ny})"
    else:
        cells_per_task = int(df_subset["cells_per_task"].iloc[0])
        subtitle = f"(Local Grid: {cells_per_task:,} Cells/Task)"

    # Plot metrics
    ax.plot(df_subset["tasks"], df_subset["runtime"], marker="o", linewidth=2, label="Runtime")
    ax.plot(df_subset["tasks"], df_subset["compute_time"], marker="s", linewidth=2, label="Compute time")
    ax.plot(df_subset["tasks"], df_subset["comm_time"], marker="^", linewidth=2, label="Comm time")
    
    # Styling
    ax.set_title(f"{scaling_type} Scaling ({dim_str})\n{subtitle}", fontsize=12)
    ax.set_xlabel("Tasks [count]")
    ax.set_ylabel("Time [s]")
    format_axes(ax, df_subset["tasks"].unique())
    ax.legend()

# Plot the 4 quadrants
plot_benchmark(axes[0, 0], strong_1d, "Strong", "1D Decomposition")
plot_benchmark(axes[0, 1], weak_1d,   "Weak",   "1D Decomposition")
plot_benchmark(axes[1, 0], strong_2d, "Strong", "2D Decomposition")
plot_benchmark(axes[1, 1], weak_2d,   "Weak",   "2D Decomposition")

# Add an explanatory text box at the bottom reflecting the A100 hardware
glossary = (
    "Hardware & Units:\n"
    "s = seconds  |  Tasks = Number of allocated MPI processes [count] (1 GPU per task)\n"
    "Runtime = Total wall-clock time  |  Compute time = NVIDIA A100 GPU calculation  |  Comm time = bwUniCluster MPI network overhead using Host Buffers\n"
    "1D Decomposition = Slicing along the Y-axis only  |  2D Decomposition = Slicing along X and Y axes (Grid layout)"
)
fig.text(0.5, 0.02, glossary, ha='center', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# Adjust layouts to accommodate titles, spacing between rows, and the bottom glossary
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.35) 

plt.savefig("benchmarks_a100_short_1d_2d.png", dpi=600)
plt.show()
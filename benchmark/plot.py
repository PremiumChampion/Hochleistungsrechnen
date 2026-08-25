import pandas as pd
import matplotlib.pyplot as plt
import math as math
from matplotlib.ticker import FixedLocator, FuncFormatter

# Load data
df = pd.read_csv("benchmark_results_gpu_a100_short_v2.csv")

# Filter into 4 separate dataframes
strong_1d = df[(df["scaling_type"] == "strong") & (df["dim"] == 1)].sort_values("tasks")
weak_1d   = df[(df["scaling_type"] == "weak")   & (df["dim"] == 1)].sort_values("tasks")
strong_2d = df[(df["scaling_type"] == "strong") & (df["dim"] == 2)].sort_values("tasks")
weak_2d   = df[(df["scaling_type"] == "weak")   & (df["dim"] == 2)].sort_values("tasks")

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
    
    if df_subset.empty:
        ax.set_title(f"{scaling_type} Scaling ({dim_str})\n(No data available)", fontsize=12)
        ax.text(0.5, 0.5, 'Incomplete/No Data', horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes, color='gray')
        ax.set_xlabel("Tasks (count)")
        ax.set_ylabel("Time (s)")
        return

    # Extract problem sizes dynamically from the CSV data
    if scaling_type == "Strong":
        nx = int(df_subset["Nx"].iloc[0])
        ny = int(df_subset["Ny"].iloc[0])
        subtitle = f"(Global Grid: {nx} x {ny})"
    else:
        cells_per_task = int(math.sqrt(int(df_subset["cells_per_task"].iloc[0])))
        subtitle = f"(Local Grid: {cells_per_task} x {cells_per_task})"

    # Plot metrics
    ax.plot(df_subset["tasks"], df_subset["runtime"], marker="o", linewidth=2, label="Runtime", color="tab:blue")
    ax.plot(df_subset["tasks"], df_subset["compute_time"], marker="s", linewidth=2, label="Compute time", color="tab:orange")
    ax.plot(df_subset["tasks"], df_subset["comm_time"], marker="^", linewidth=2, label="Communication time", color="tab:green")
    
    # Add Data Labels above/below the points
    for _, row in df_subset.iterrows():
        # Runtime Label (Above)
        ax.annotate(f"{row['runtime']:.1f}", (row['tasks'], row['runtime']), 
                    textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='darkblue')
        # Compute Time Label (Below, to prevent overlapping with runtime)
        ax.annotate(f"{row['compute_time']:.1f}", (row['tasks'], row['compute_time']), 
                    textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color='darkred')
        # Comm Time Label (Above)
        ax.annotate(f"{row['comm_time']:.1f}", (row['tasks'], row['comm_time']), 
                    textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='darkgreen')

    # Styling
    ax.set_title(f"{scaling_type} Scaling ({dim_str})\n{subtitle}", fontsize=12)
    ax.set_xlabel("Tasks (count)")
    ax.set_ylabel("Time (s)")
    format_axes(ax, df_subset["tasks"].unique())
    ax.legend()

def create_figure(strong_df, weak_df, dim_str, output_filename):
    """Creates a 1x2 figure for a specific decomposition type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # Main title
    fig.suptitle(f"Fluid Simulation Scaling Benchmarks: bwUniCluster (gpu_a100_short)\n{dim_str}", 
                 fontsize=16, fontweight='bold')
    
    # Plot both sides
    plot_benchmark(axes[0], strong_df, "Strong", dim_str)
    plot_benchmark(axes[1], weak_df, "Weak", dim_str)
    
    # Set proper glossary text based on dimension
    if "1D" in dim_str:
        decomp_desc = "1D Decomposition = Slicing along the Y-axis only"
    else:
        decomp_desc = "2D Decomposition = Slicing along X and Y axes (Grid layout)"

    glossary = (
        "Hardware & Units:\n"
        "s = seconds  |  Tasks = Number of allocated MPI processes (count) (1 GPU per task)\n"
        "Runtime = Total wall-clock time  |  Compute time = NVIDIA A100 GPU calculation  |  Communication time = copy overhead\n"
        f"{decomp_desc}"
    )
    
    fig.text(0.5, 0.02, glossary, ha='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # Adjust layouts 
    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.20, wspace=0.15) 

    plt.savefig(output_filename, dpi=600)
    print(f"Saved: {output_filename}")
    plt.show()

# ---------------------------------------------------------
# Generate the isolated graphs
# ---------------------------------------------------------

# 1. Plot 1D Benchmarks
create_figure(strong_1d, weak_1d, "1D Decomposition", "benchmarks_a100_short_1d.png")

# 2. Plot 2D Benchmarks
create_figure(strong_2d, weak_2d, "2D Decomposition", "benchmarks_a100_short_2d.png")
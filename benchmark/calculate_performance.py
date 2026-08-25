import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

def plot_performance_glups_linear(csv_file="benchmark_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # 1. Load Data and calculate GLUPS
    df = pd.read_csv(csv_file)
    
    # Calculate MLUPS if missing, then convert to GLUPS
    if 'mlups' not in df.columns:
        df['mlups'] = (df['Nx'] * df['Ny'] * df['steps']) / (df['runtime'] * 1e6)
    df['glups'] = df['mlups'] / 1000.0

    # 2. Setup Figure (1x2 Subplots: Strong Scaling & Weak Scaling)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Fluid Simulation Performance Scaling on NVIDIA A100", 
                 fontsize=16, fontweight='bold')

    # Styles for the plot
    styles = {
        1: ('tab:blue', 'o', '1D Decomposition'),
        2: ('tab:orange', 's', '2D Decomposition')
    }

    # =========================================================
    # Subplot 1: Strong Scaling
    # =========================================================
    ax1 = axes[0]
    strong_df = df[df['scaling_type'] == 'strong'].sort_values('tasks')
    
    for dim in [1, 2]:
        subset = strong_df[strong_df['dim'] == dim]
        if not subset.empty:
            color, marker, label = styles[dim]
            
            # Actual Data
            ax1.plot(subset['tasks'], subset['glups'], marker=marker, 
                     color=color, linewidth=2.5, markersize=7, label=label)
            
            # Ideal Scaling Line (linear scaling from 1 GPU)
            single_task = subset[subset['tasks'] == 1]
            if not single_task.empty:
                base_glups = single_task['glups'].iloc[0]
                # Include 0 in the ideal line so it draws straight to the origin
                ideal_tasks = [0] + list(subset['tasks'])
                ideal_glups = [0] + [base_glups * t for t in subset['tasks']]
                ax1.plot(ideal_tasks, ideal_glups, linestyle='--', 
                         color=color, alpha=0.6, label=f'Ideal Scaling ({label})')
                
            # Add data labels
            for _, row in subset.iterrows():
                # Stagger labels slightly based on dimension to prevent overlap
                y_offset = -12 if dim == 1 else 8
                ax1.annotate(f"{row['glups']:.2f}", (row['tasks'], row['glups']), 
                             textcoords="offset points", xytext=(0, y_offset), 
                             ha='center', fontsize=8, color=color)

    ax1.set_title("Strong Scaling\n(Fixed Global Grid)", fontsize=13)
    ax1.set_xlabel("Number of GPUs (Tasks)", fontsize=11)
    ax1.set_ylabel("Performance (GLUPS)", fontsize=11)

    # =========================================================
    # Subplot 2: Weak Scaling
    # =========================================================
    ax2 = axes[1]
    weak_df = df[df['scaling_type'] == 'weak'].sort_values('tasks')
    
    for dim in [1, 2]:
        subset = weak_df[weak_df['dim'] == dim]
        if not subset.empty:
            color, marker, label = styles[dim]
            
            # Actual Data
            ax2.plot(subset['tasks'], subset['glups'], marker=marker, 
                     color=color, linewidth=2.5, markersize=7, label=label)
            
            # Ideal Scaling Line (linear scaling from 1 GPU)
            single_task = subset[subset['tasks'] == 1]
            if not single_task.empty:
                base_glups = single_task['glups'].iloc[0]
                ideal_tasks = [0] + list(subset['tasks'])
                ideal_glups = [0] + [base_glups * t for t in subset['tasks']]
                ax2.plot(ideal_tasks, ideal_glups, linestyle='--', 
                         color=color, alpha=0.6, label=f'Ideal Scaling ({label})')
                
            # Add data labels
            for _, row in subset.iterrows():
                y_offset = -12 if dim == 1 else 8
                ax2.annotate(f"{row['glups']:.2f}", (row['tasks'], row['glups']), 
                             textcoords="offset points", xytext=(0, y_offset), 
                             ha='center', fontsize=8, color=color)

    ax2.set_title("Weak Scaling\n(Grid Grows Proportionally with GPUs)", fontsize=13)
    ax2.set_xlabel("Number of GPUs (Tasks)", fontsize=11)
    ax2.set_ylabel("Performance (GLUPS)", fontsize=11)

    # =========================================================
    # Formatting for both axes (The changes for including 0)
    # =========================================================
    for ax in [ax1, ax2]:
        # Force the X and Y axes to start at exactly 0
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Explicitly set X-ticks so that 0, 1, 2, 4, 8, 16 show up nicely
        custom_ticks = [0, 1, 2, 4, 8, 12, 16] 
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([str(t) for t in custom_ticks])
            
        ax.grid(True, which="major", linestyle="-", alpha=0.35)
        ax.legend(loc='upper left', fontsize=9)

    # =========================================================
    # Glossary / Info Box
    # =========================================================
    glossary = (
        "Metrics & Hardware:\n"
        "GLUPS = Giga Lattice Updates Per Second (Higher is better)  |  1 GPU per Task\n"
        "Ideal Scaling = 100% parallel efficiency (linear growth based on a single GPU's performance)"
    )
    fig.text(0.5, 0.02, glossary, ha='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.15)
    
    # Save output
    out_fn = "performance_glups_linear.png"
    plt.savefig(out_fn, dpi=600)
    print(f"Successfully generated plot: {out_fn}")
    plt.show()

if __name__ == "__main__":
    plot_performance_glups_linear("benchmark_results_gpu_a100_short_v2.csv")
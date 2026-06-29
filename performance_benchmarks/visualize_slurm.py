import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

def generate_total_cores_plot(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(csv_file)
    df['Nodes'] = pd.to_numeric(df['Nodes'])
    df['Cores per node'] = pd.to_numeric(df['Cores per node'])
    df['Runtime (s)'] = pd.to_numeric(df['Runtime (s)'])
    df['Performance (MLUPS)'] = pd.to_numeric(df['Performance (MLUPS)'])
    
    # Calculate Total Cores (the requested X-axis)
    df['Total Cores'] = df['Nodes'] * df['Cores per node']
    
    # Sort by total cores to ensure lines connect points in order
    df = df.sort_values('Total Cores')

    # 2. Setup Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx() # Create the second Y-axis

    # Style definitions
    # We use a distinct color for each node count
    unique_nodes = sorted(df['Nodes'].unique())
    # Use a professional color palette
    colors = plt.cm.tab10(range(len(unique_nodes))) 
    
    node_handles = []

    # 3. Plotting
    for i, node_count in enumerate(unique_nodes):
        node_data = df[df['Nodes'] == node_count]
        color = colors[i]
        
        # Plot Runtime on Left Axis (Solid Line)
        ln1, = ax1.plot(node_data['Total Cores'], node_data['Runtime (s)'], 
                        marker='o', linestyle='-', color=color, linewidth=2, 
                        label=f'{node_count} Node(s)')
        
        # Plot Performance on Right Axis (Dashed Line)
        ax2.plot(node_data['Total Cores'], node_data['Performance (MLUPS)'], 
                 marker='s', linestyle='--', color=color, linewidth=2, alpha=0.7)
        
        node_handles.append(ln1)

    # 4. Axes and Grid Styling
    ax1.set_xlabel('Total Cores (Nodes × Cores per Node)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (s)', color='#1f77b4', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance (MLUPS)', color='#d62728', fontsize=12, fontweight='bold')
    
    # Match tick colors to their respective axis colors
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    plt.title('Fluid Simulation Scaling Analysis\nRuntime and Performance vs. Total Cores Assigned', 
              pad=20, fontsize=15, fontweight='bold')

    # 5. Complex Legend
    # First legend: Node Configurations (Colors)
    legend1 = ax1.legend(handles=node_handles, title="Configurations", 
                         loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                         ncol=len(unique_nodes), frameon=True, shadow=True)
    ax1.add_artist(legend1)

    # Second legend: Metric types (Line styles)
    runtime_line = mlines.Line2D([], [], color='gray', marker='o', linestyle='-', label='Runtime (s)')
    perf_line = mlines.Line2D([], [], color='gray', marker='s', linestyle='--', label='Performance (MLUPS)')
    ax2.legend(handles=[runtime_line, perf_line], title="Metrics", 
               loc='upper right', frameon=True, shadow=True)

    # Adjust layout to fit legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # 6. Save and Show
    output_fn = "scaling_total_cores.png"
    plt.savefig(output_fn, dpi=300)
    print(f"Generated combined plot: {output_fn}")
    plt.show()

if __name__ == "__main__":
    generate_total_cores_plot("slurm_summary.csv")
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_slurm_files(directory="."):
    data = []
    
    # Regex patterns based on your file format
    patterns = {
        "nodes": re.compile(r"Running on (\d+) nodes"),
        "cores_per_node": re.compile(r"with (\d+) cores each"),
        "ranks": re.compile(r"distributed over (\d+) ranks"),
        "runtime": re.compile(r"Runtime:\s+([\d.]+)\s+s"),
        "performance": re.compile(r"Performance:\s+([\d.]+)\s+MLUPS"),
        "walltime": re.compile(r"Job Wall-clock time:\s+(\d+:\d+:\d+)")
    }

    for filename in os.listdir(directory):
        if filename.startswith("slurm-") and filename.endswith(".out"):
            filepath = os.path.join(directory, filename)
            file_info = {"filename": filename}
            
            with open(filepath, 'r') as f:
                content = f.read()
                
                for key, pattern in patterns.items():
                    match = pattern.search(content)
                    if match:
                        file_info[key] = match.group(1)
            
            # Check if we found the essential metrics before adding
            if "runtime" in file_info and "performance" in file_info:
                # Convert numeric types
                file_info["nodes"] = int(file_info.get("nodes", 0))
                file_info["cores_per_node"] = int(file_info.get("cores_per_node", 0))
                file_info["ranks"] = int(file_info.get("ranks", 0))
                file_info["total_cores"] = file_info["nodes"] * file_info["cores_per_node"]
                file_info["runtime"] = float(file_info["runtime"])
                file_info["performance"] = float(file_info["performance"])
                
                data.append(file_info)

    return pd.DataFrame(data)

def generate_plots(df):
    if df.empty:
        print("No data found to plot.")
        return

    # Sort data by total cores for logical line plotting
    df = df.sort_values("total_cores")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot Runtime (Left Axis) ---
    color_runtime = 'tab:blue'
    ax1.set_xlabel('Total Computational Resources (Total Cores)')
    ax1.set_ylabel('Runtime (seconds)', color=color_runtime, fontsize=12)
    line1 = ax1.plot(df["total_cores"], df["runtime"], marker='o', color=color_runtime, label='Runtime (s)')
    ax1.tick_params(axis='y', labelcolor=color_runtime)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Plot Performance (Right Axis) ---
    ax2 = ax1.twinx()  
    color_perf = 'tab:red'
    ax2.set_ylabel('Performance (MLUPS)', color=color_perf, fontsize=12)
    line2 = ax2.plot(df["total_cores"], df["performance"], marker='s', color=color_perf, label='Performance (MLUPS)')
    ax2.tick_params(axis='y', labelcolor=color_perf)

    # --- Formatting ---
    plt.title('Milestone 06: Scaling Analysis\n(Runtime vs. Performance)', fontsize=14)
    
    # Combined legend
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center')

    # Add text annotations for Ranks/Nodes at data points
    for i, row in df.iterrows():
        ax1.annotate(f'R:{int(row["ranks"])}', 
                     (row["total_cores"], row["runtime"]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    fig.tight_layout()
    plt.savefig('scaling_performance.png', dpi=300)
    print("Figure saved as scaling_performance.png")

if __name__ == "__main__":
    # 1. Extract Data
    results_df = parse_slurm_files()
    
    if not results_df.empty:
        # 2. Save to CSV
        csv_name = "scaling_results.csv"
        results_df.to_csv(csv_name, index=False)
        print(f"Data exported to {csv_name}")
        
        # 3. Print summary to console
        print("\nParsed Data Summary:")
        print(results_df[["nodes", "total_cores", "ranks", "runtime", "performance"]])
        
        # 4. Generate Figures
        generate_plots(results_df)
    else:
        print("No valid slurm-xxx.out files found in the current directory.")
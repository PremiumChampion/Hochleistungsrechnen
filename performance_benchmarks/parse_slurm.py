import re
import csv
import glob
import os

def parse_slurm_files(directory_path, output_csv):
    # Define the regex patterns for each field
    patterns = {
        "Total Mass": r"Total mass:\s+([\d.e+-]+)",
        "Kinetic Energy": r"Total kinetic energy:\s+([\d.]+)",
        "Runtime (s)": r"Runtime:\s+([\d.]+)\s+s",
        "Performance (MLUPS)": r"Performance:\s+([\d.]+)\s+MLUPS",
        "Real Time": r"real\s+(\d+m[\d.]+s)",
        "User Time": r"user\s+(\d+m[\d.]+s)",
        "Sys Time": r"sys\s+(\d+m[\d.]+s)",
        "Nodes": r"Nodes:\s+(\d+)",
        "Cores per node": r"Cores per node:\s+(\d+)",
        "CPU Utilized": r"CPU Utilized:\s+([\d:]+)",
        "CPU Efficiency": r"CPU Efficiency:\s+([\d.]+% of [\d:]+ core-walltime)",
        "Wall-clock time": r"Job Wall-clock time:\s+([\d:]+)",
        "State": r"State:\s+(.*)"
    }

    # Prepare the list to hold all extracted data
    data_rows = []

    # Find all files matching the pattern
    files = glob.glob(os.path.join(directory_path, "slurm-*.out"))
    
    if not files:
        print("No files found matching slurm-*.out")
        return

    print(f"Processing {len(files)} files...")

    for file_path in files:
        # Initialize a dictionary for this file with None values
        file_data = {key: None for key in patterns.keys()}
        file_data["Filename"] = os.path.basename(file_path)

        with open(file_path, 'r') as f:
            content = f.read()
            # We search the whole content to handle multiline or out-of-order logs
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    file_data[key] = match.group(1)
        
        data_rows.append(file_data)

    # Write to CSV
    headers = ["Filename"] + list(patterns.keys())
    
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_rows)
        print(f"Successfully saved data to {output_csv}")
    except IOError as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    # Configuration: Change '.' to your directory path if needed
    TARGET_DIRECTORY = "." 
    OUTPUT_FILE = "slurm_summary.csv"
    
    parse_slurm_files(TARGET_DIRECTORY, OUTPUT_FILE)
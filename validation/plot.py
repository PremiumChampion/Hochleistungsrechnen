import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
df_mass = pd.read_csv("mass_conservation.csv")
df_relax = pd.read_csv("relaxation.csv")

# 2. Calculate Error Metrics
consecutive_diffs = df_mass["total_mass"].diff().abs()
accumulated_diff_series = consecutive_diffs.cumsum().fillna(0)

# Extract scalar metrics
total_accumulated = consecutive_diffs.sum()
max_single_jump = consecutive_diffs.max()
avg_drift = consecutive_diffs.mean()

# 3. Create a figure with a 2D left subplot and a 3D right subplot
fig = plt.figure(figsize=(14, 6))
fig.suptitle("LBM Physics Validation", fontsize=16, fontweight='bold')

# ---------------------------------------------------------
# Plot 1: Global Mass Conservation & Error Tracking (2D)
# ---------------------------------------------------------
ax1 = fig.add_subplot(1, 2, 1)

# Plot mass over time on the Primary Y-Axis (Left)
line1 = ax1.plot(df_mass["step"], df_mass["total_mass"], color="tab:blue", linewidth=2.5, label="Total Mass")
ax1.set_title("Global Mass Conservation", fontsize=13)
ax1.set_xlabel("Time Step (ts)", fontsize=11)
ax1.set_ylabel(r"Total Domain Mass ($\sum \rho$) (mu)", fontsize=11, color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, linestyle="--", alpha=0.5)

# Force the y-axis to not use scientific offset so we can visually confirm precision
ax1.ticklabel_format(useOffset=False, style='plain')
initial_mass = df_mass["total_mass"].iloc[0]
ax1.set_ylim(initial_mass - 1e-11, initial_mass + 1e-11)

# Plot accumulated error on the Secondary Y-Axis (Right)
ax_err = ax1.twinx()
line2 = ax_err.plot(df_mass["step"], accumulated_diff_series, color="tab:red", linewidth=2, linestyle="--", label="Accumulated Error")
ax_err.set_ylabel("Accumulated Error (mu)", fontsize=11, color="tab:red")
ax_err.tick_params(axis='y', labelcolor="tab:red")

# Combine legends for both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

# Add the Metrics Text Box
stats_text = (
    f"Numerical Round-off Error:\n"
    f"• Total Accumulated: {total_accumulated:.3e} mu\n"
    f"• Max Single Jump: {max_single_jump:.3e} mu\n"
    f"• Average Drift/Step: {avg_drift:.3e} mu"
)
ax1.text(0.03, 0.96, stats_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='tab:red'))

# ---------------------------------------------------------
# Plot 2: Density Relaxation (3D Waterfall Diagram)
# ---------------------------------------------------------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Plot EVERY time step available in the dataset so they don't disappear
unique_t = sorted(df_relax['t'].unique())
colors = plt.cm.rainbow_r(np.linspace(0, 1, len(unique_t)))

for t, color in zip(unique_t, colors):
    group = df_relax[df_relax['t'] == t]
    
    # In 3D: X is position, Y is time step, Z is density
    ax2.plot(group["x"], np.full_like(group["x"], t), group["rho"], color=color, linewidth=2, alpha=0.85)

ax2.set_title("Density Relaxation (Damped Acoustic Waves)", fontsize=13)
ax2.set_xlabel("x Position (lu)", fontsize=11, labelpad=10)
ax2.set_ylabel("Time Step (ts)", fontsize=11, labelpad=10)
ax2.set_zlabel(r"Local Density ($\rho$) (mu/lu$^2$)", fontsize=11, labelpad=10)

# Force the axis ticks to display every 25 steps (visually)
max_t = int(max(unique_t)) if len(unique_t) > 0 else 200
ax2.set_yticks(np.arange(0, max_t + 1, 25))

# Invert the time step axis (Y-Axis in this 3D mapping)
ax2.invert_yaxis()

# Adjust viewing angle for better 3D waterfall perspective
ax2.view_init(elev=25, azim=-55)

# ---------------------------------------------------------
# Formatting and Glossary
# ---------------------------------------------------------
glossary = (
    "Physics Validation & Units:\n"
    "lu = lattice units (distance)  |  ts = time steps (time)  |  mu = mass unit"
)
fig.text(0.5, 0.02, glossary, ha='center', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# Adjust layouts to make room for glossary
plt.subplots_adjust(left=0.09, right=0.95, top=0.88, bottom=0.15, wspace=0.2)

# Save and Show
plt.savefig("physics_validation_3d.png", dpi=600)
plt.show()
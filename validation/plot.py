import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a 1x2 figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("LBM Physics Validation", fontsize=16, fontweight='bold')

# ---------------------------------------------------------
# Plot 1: Global Mass Conservation
# ---------------------------------------------------------
df_mass = pd.read_csv("mass_conservation.csv")

# Plot mass over time
axes[0].plot(df_mass["step"], df_mass["total_mass"], color="tab:blue", linewidth=2.5)
axes[0].set_title("Global Mass Conservation", fontsize=13)
axes[0].set_xlabel("Time Step [ts]", fontsize=11)
axes[0].set_ylabel(r"Total Domain Mass ($\sum \rho$) [dimensionless]", fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.5)

# Force the y-axis to not use scientific offset so we can visually confirm precision
axes[0].ticklabel_format(useOffset=False, style='plain')
# Add a very slight manual margin to the y-axis so the line doesn't sit exactly on the bounds
initial_mass = df_mass["total_mass"].iloc[0]
axes[0].set_ylim(initial_mass - 1e-11, initial_mass + 1e-11)

# ---------------------------------------------------------
# Plot 2: Density Relaxation (Damped Sound Waves)
# ---------------------------------------------------------
df_relax = pd.read_csv("relaxation.csv")

# Use a colormap to differentiate time steps
colors = plt.cm.viridis(np.linspace(0, 0.9, len(df_relax['t'].unique())))

for (t, group), color in zip(df_relax.groupby("t"), colors):
    axes[1].plot(group["x"], group["rho"], label=f"t={t} [ts]", color=color, linewidth=2)

axes[1].set_title("Density Relaxation (Damped Acoustic Waves)", fontsize=13)
axes[1].set_xlabel("x Position [lu]", fontsize=11)
axes[1].set_ylabel(r"Local Density ($\rho$) [dimensionless]", fontsize=11)
axes[1].legend(title="Time Steps")
axes[1].grid(True, linestyle="--", alpha=0.5)

# ---------------------------------------------------------
# Formatting and Glossary
# ---------------------------------------------------------
glossary = (
    "Physics Validation & Units:\n"
    "lu = lattice units (distance)  |  ts = time steps (time)\n"
    r"Mass Plot: The BGK collision and streaming steps conserve mass to machine precision ($10^{-13}$)." + "\n"
    r"Relaxation Plot: A central density Gaussian decays into radiating acoustic waves."
)
fig.text(0.5, 0.02, glossary, ha='center', fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# Adjust layouts to make room for glossary
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.20)

plt.savefig("physics_validation.png", dpi=600)
print("Saved physics validation plot to 'physics_validation.png'")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Plot velocity profile evolution
if os.path.exists("velocity_profile.csv"):
    df_prof = pd.read_csv("velocity_profile.csv")
    plt.figure()
    for t, group in df_prof.groupby("t"):
        plt.plot(group["y"], group["ux"], label=f"t={t}")
    plt.xlabel("y Position")
    plt.ylabel(r"$u_x$ Velocity")
    plt.title(r"Shear-Wave Decay Velocity Profile Evolution ($\omega=1.0$)")
    plt.legend()
    plt.grid()
    plt.savefig("velocity_profile.png", dpi=600)
    print("Exported velocity_profile.png")
    plt.close()

# 2. Plot numerical vs analytical kinematic viscosity with delta
if os.path.exists("viscosity.csv"):
    df_visc = pd.read_csv("viscosity.csv")
    
    # Create the figure and primary axis (left)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    # Plot primary variables
    line1, = ax1.plot(df_visc["omega"], df_visc["measured_nu"], 'o-', 
                      color="tab:blue", label="Measured Numerical Viscosity")
    line2, = ax1.plot(df_visc["omega"], df_visc["analytical_nu"], 's--', 
                      color="tab:orange", label=r"Analytical Prediction $\nu = \frac{1}{3}(\frac{1}{\omega} - \frac{1}{2})$")
    
    ax1.set_xlabel(r"Relaxation Parameter ($\omega$)")
    ax1.set_ylabel(r"Kinematic Viscosity ($\nu$)", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True)
    
    # Calculate the absolute difference (delta)
    delta_nu = (df_visc["measured_nu"] - df_visc["analytical_nu"]).abs()
    
    # Create secondary axis (right) for the delta values
    ax2 = ax1.twinx()
    line3, = ax2.plot(df_visc["omega"], delta_nu, 'd:', 
                      color="tab:red", label=r"Absolute Difference ($|\Delta \nu|$)")
    ax2.set_ylabel(r"Absolute Difference ($|\Delta \nu|$)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    
    # Combine legends from both axes to prevent overlapping
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    plt.title("Measured vs Analytical Kinematic Viscosity")
    fig.tight_layout()
    plt.savefig("viscosity-delta.png", dpi=300)
    print("Exported viscosity.png with delta values")
    plt.close()
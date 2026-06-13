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
    plt.ylabel("$u_x$ Velocity")
    plt.title("Shear-Wave Decay Velocity Profile Evolution ($\omega=1.0$)")
    plt.legend()
    plt.grid()
    plt.savefig("velocity_profile.png")
    print("Exported velocity_profile.png")
    plt.close()

# 2. Plot numerical vs analytical kinematic viscosity
if os.path.exists("viscosity.csv"):
    df_visc = pd.read_csv("viscosity.csv")
    plt.figure()
    plt.plot(df_visc["omega"], df_visc["measured_nu"], 'o-', label="Measured Numerical Viscosity")
    plt.plot(df_visc["omega"], df_visc["analytical_nu"], 's--', label="Analytical Prediction $\\nu = \\frac{1}{3}(\\frac{1}{\\omega} - \\frac{1}{2})$")
    plt.xlabel("Relaxation Parameter ($\omega$)")
    plt.ylabel("Kinematic Viscosity ($\\nu$)")
    plt.title("Measured vs Analytical Kinematic Viscosity")
    plt.legend()
    plt.grid()
    plt.savefig("viscosity.png")
    print("Exported viscosity.png")
    plt.close()
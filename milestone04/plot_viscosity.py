import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# 1. Plot velocity profile evolution and analytical error overlay
if os.path.exists("velocity_profile.csv"):
    df_prof = pd.read_csv("velocity_profile.csv")
    
    # Create 1x2 subplots: Left for profiles, Right for error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    fig.suptitle(r"Shear-Wave Decay Validation ($\omega=1.0$)", fontsize=14, fontweight='bold')
    
    # Extract grid parameters and calculate physics constants
    Ny = df_prof["y"].max() + 1
    omega = 1.0
    nu = (1.0 / 3.0) * (1.0 / omega - 0.5)
    k = 2.0 * np.pi / Ny
    u0 = 0.05  # Initial amplitude defined in C++ (epsilon)
    
    time_handles = []
    
    for t, group in df_prof.groupby("t"):
        # Plot Numerical Simulation Data as scatter points
        p, = ax1.plot(group["y"], group["ux"], 'o', markersize=4, alpha=0.8)
        color = p.get_color()
        
        # Save a handle for the legend
        time_handles.append(mlines.Line2D([], [], color=color, marker='o', linestyle='-', label=f"t={t} [ts]"))
        
        # Calculate Analytical Solution (smooth curve for left plot)
        y_vals_smooth = np.linspace(0, Ny - 1, 200)
        u_ana_smooth = u0 * np.exp(-nu * (k**2) * t) * np.sin(k * y_vals_smooth)
        ax1.plot(y_vals_smooth, u_ana_smooth, '-', color=color, linewidth=1.5)
        
        # Calculate exact Analytical Error at grid nodes
        u_ana_discrete = u0 * np.exp(-nu * (k**2) * t) * np.sin(k * group["y"])
        error = group["ux"] - u_ana_discrete
        
        # Plot error scaled by 10^8 to remove the floating scientific notation multiplier
        ax2.plot(group["y"], error * 1e8, '-', color=color, linewidth=1.5)

    # Style Left Subplot (Profiles)
    ax1.set_xlabel("y Position [lu]")
    ax1.set_ylabel(r"$u_x$ Velocity [lu/ts]")
    ax1.set_title("Velocity Profile Evolution")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Custom Legend
    sim_proxy = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=5, label='Simulation')
    ana_proxy = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2, label='Analytical')
    ax1.legend(handles=[sim_proxy, ana_proxy] + time_handles, loc='upper right', ncol=2, fontsize=9)
    
    # Style Right Subplot (Error)
    ax2.set_xlabel("y Position [lu]")
    # Update label to reflect the scale and the correct LBM units
    ax2.set_ylabel(r"Error ($u_{\mathrm{num}} - u_{\mathrm{ana}}$) [$10^{-8}$ lu/ts]")
    ax2.set_title("Absolute Analytical Error")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # Add an explanatory text box at the bottom
    glossary = (
        "Abbreviations & Units:\n"
        "lu = lattice units (distance)  |  ts = time steps (time)\n"
        r"$u_x$ = horizontal velocity  |  $t$ = simulation time  |  $\omega$ = relaxation parameter"
    )
    fig.text(0.5, 0.02, glossary, ha='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Adjust layout to fit titles and glossary box
    plt.subplots_adjust(top=0.88, bottom=0.22, wspace=0.2) 
    
    plt.savefig("velocity_profile.png", dpi=600)
    print("Exported velocity_profile.png with analytical error overlay")
    plt.close()


# 2. Plot numerical vs analytical kinematic viscosity with delta
if os.path.exists("viscosity.csv"):
    df_visc = pd.read_csv("viscosity.csv")
    
    fig, ax1 = plt.subplots(figsize=(8, 6.5))
    
    line1, = ax1.plot(df_visc["omega"], df_visc["measured_nu"], 'o-', 
                      color="tab:blue", label="Measured Numerical Viscosity")
    line2, = ax1.plot(df_visc["omega"], df_visc["analytical_nu"], 's--', 
                      color="tab:orange", label=r"Analytical Prediction $\nu = \frac{1}{3}(\frac{1}{\omega} - \frac{1}{2})$")
    
    ax1.set_xlabel(r"Relaxation Parameter ($\omega$) [dimensionless]")
    ax1.set_ylabel(r"Kinematic Viscosity ($\nu$) [lu$^2$/ts]", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True)
    
    delta_nu = (df_visc["measured_nu"] - df_visc["analytical_nu"]).abs()
    
    ax2 = ax1.twinx()
    line3, = ax2.plot(df_visc["omega"], delta_nu, 'd:', 
                      color="tab:red", label=r"Absolute Difference ($|\Delta \nu|$)")
    ax2.set_ylabel(r"Absolute Difference ($|\Delta \nu|$) [lu$^2$/ts]", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    plt.title("Measured vs Analytical Kinematic Viscosity")
    
    glossary = (
        "Abbreviations & Units:\n"
        "lu = lattice units (distance)  |  ts = time steps (time)\n"
        r"$\omega$ = relaxation parameter  |  $\nu$ = kinematic viscosity  |  $\Delta \nu$ = error"
    )
    fig.text(0.5, 0.02, glossary, ha='center', fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.subplots_adjust(bottom=0.22)
    
    plt.savefig("viscosity-delta.png", dpi=300)
    print("Exported viscosity-delta.png")
    plt.close()
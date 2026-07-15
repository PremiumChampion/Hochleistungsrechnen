import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Plot lid-driven cavity streamlines
if os.path.exists("lid_driven_cavity.csv"):
    df = pd.read_csv("lid_driven_cavity.csv")
    
    X = np.sort(df['x'].unique())
    Y = np.sort(df['y'].unique())
    
    U = np.zeros((len(Y), len(X)))
    V = np.zeros((len(Y), len(X)))
    
    for index, row in df.iterrows():
        xi = int(row['x'])
        yi = int(row['y'])
        U[yi, xi] = row['ux']
        V[yi, xi] = row['uy']
        
    speed = np.sqrt(U**2 + V**2)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    strm = ax.streamplot(X, Y, U, V, color=speed, cmap='viridis', density=1.5)
    fig.colorbar(strm.lines, ax=ax, label='Velocity magnitude [lu/ts]')
    
    ax.set_title("Lid-driven cavity flow streamlines")
    ax.set_xlabel("x Position [lu]")
    ax.set_ylabel("y Position [lu]")
    ax.set_aspect('equal')
    
    glossary = (
        "Abbreviations & Units:\n"
        "lu = lattice units (distance)  |  ts = time steps (time)\n"
        "x, y = grid coordinates  |  u, v = horizontal & vertical velocity components"
    )
    fig.text(0.5, 0.02, glossary, ha='center', fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.subplots_adjust(bottom=0.18)
    
    plt.savefig("lid_driven_cavity.png", dpi=600)
    print("Exported lid_driven_cavity.png")
    plt.close()

# 2. Plot centerline velocity profile
if os.path.exists("centerline_profile.csv"):
    df_prof = pd.read_csv("centerline_profile.csv")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    L = df_prof['y'].max()
    
    # Calculate percentages: (position / max_length) * 100
    y_norm = (df_prof['y'] / L) * 100
    
    # Calculate percentages: (velocity / lid_velocity) * 100
    u_norm = (df_prof['ux'] / 0.1) * 100
    
    ax.plot(y_norm, u_norm, label="Simulation", color='blue')
    
    # Update labels to indicate percentages
    ax.set_xlabel("$y / L$ [%]")
    ax.set_ylabel("$u_x / u_{\mathrm{lid}}$ [%]")
    ax.set_title("Centerline Velocity Profile ($x = L/2$)")
    ax.legend()
    ax.grid()
    
    glossary = (
        "Abbreviations & Base Units:\n"
        "lu = lattice units (dist)  |  ts = time steps (time)  |  $L$ = total domain height [lu]\n"
        "$u_{\mathrm{lid}}$ = driving lid velocity [lu/ts]  |  $u_x$ = horizontal fluid velocity [lu/ts]  |  $y$ = vertical position [lu]"
    )
    fig.text(0.5, 0.02, glossary, ha='center', fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.subplots_adjust(bottom=0.18)
    
    plt.savefig("centerline_profile.png", dpi=600)
    print("Exported centerline_profile.png")
    plt.close()
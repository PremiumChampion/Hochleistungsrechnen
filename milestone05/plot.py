import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    strm = ax.streamplot(X, Y, U, V, color=speed, cmap='viridis', density=1.5)
    fig.colorbar(strm.lines, ax=ax, label='Velocity magnitude')
    
    ax.set_title("Lid-driven cavity flow streamlines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    
    plt.savefig("lid_driven_cavity.png", dpi=150)
    print("Exported lid_driven_cavity.png")
    plt.close()

if os.path.exists("centerline_profile.csv"):
    df_prof = pd.read_csv("centerline_profile.csv")
    plt.figure()
    
    L = df_prof['y'].max()
    y_norm = df_prof['y'] / L
    
    # Normalizing velocity relative to the initial sliding lid momentum
    u_norm = df_prof['ux'] / 0.1
    
    plt.plot(y_norm, u_norm, label="Simulation", color='blue')
    plt.xlabel("$y / L$")
    plt.ylabel("$u_x / u_{\mathrm{lid}}$")
    plt.title("Centerline Velocity Profile ($x = L/2$)")
    plt.legend()
    plt.grid()
    plt.savefig("centerline_profile.png", dpi=150)
    print("Exported centerline_profile.png")
    plt.close()
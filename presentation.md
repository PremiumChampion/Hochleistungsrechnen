This is a fantastic project! You have successfully implemented a high-performance Lattice Boltzmann Method (LBM) solver with advanced features: **Kokkos** for performance portability (GPU/CPU), **MPI** for 1D/2D domain decomposition, and even a real-time **SDL3 interactive canvas** for the "entertainment" factor.

Here is a comprehensive 10-15 minute presentation schema tailored to the project requirements (Explain, Convince, Demonstrate, Entertain), along with LaTeX/TikZ code for conceptual graphics and a Python script for presentation-ready scaling plots.

---

### 📊 Presentation Schema (10–15 Minutes)

**Title:** High-Performance LBM Fluid Simulation with Kokkos & MPI

#### 1. Introduction: What the code does (2 mins)
*   **Concept:** Briefly explain the Lattice Boltzmann Method (LBM). Instead of solving Navier-Stokes macroscopically, we simulate fictitious particle distributions on a lattice (D2Q9 model).
*   **The Tech Stack:** 
    *   **C++ & Kokkos:** For executing kernels on any hardware backend (GPUs, multicore CPUs) without changing the core logic.
    *   **MPI:** For distributed memory scaling (splitting the grid across multiple nodes).
    *   **SDL3 / FFmpeg:** For real-time interactivity and video export.
*   **Visual:** Show the **D2Q9 Lattice diagram** (use the LaTeX code below).

#### 2. Convince Us: Validation & Physics (3 mins)
*   **Goal:** Prove the simulation physically makes sense.
*   **Test 1: Shear Wave Decay:** Explain how you initialized a sinusoidal velocity profile. Show the generated `velocity_profile.png` and `viscosity.png`. *Key takeaway: The measured kinematic viscosity perfectly matches the analytical prediction $v = \frac{1}{3}(\frac{1}{\omega} - \frac{1}{2})$.*
*   **Test 2: Lid-Driven Cavity:** The classic CFD benchmark. Show the `lid_driven_cavity.png` (streamlines) and `centerline_profile.png`. Mention the steady-state convergence check (max velocity change $< 10^{-6}$).

#### 3. Demonstrate: Domain Decomposition & Scaling (4 mins)
*   **The Strategy:** Explain the 2D Cartesian MPI topology. How do domains talk to each other? (Halo exchange of the boundary distribution functions).
*   **Visual:** Show the **Halo Exchange diagram** (use the LaTeX code below).
*   **Performance (MLUPS):** Define MLUPS (Mega Lattice Updates Per Second). Show your scaling graphs.
*   **Amdahl's vs. Gustafson's Law:** Discuss Strong Scaling (fixed grid, more ranks = faster time) vs. Weak Scaling (grid grows with ranks = sustained MLUPS). Mention the boundary communication overhead.

#### 4. Entertain: The Interactive Fluid Canvas (3 mins)
*   *Live Demo Time!* (Or play the FFmpeg exported video if a live demo is too risky).
*   Run your `interactive` executable.
*   Showcase drawing walls (`#`), sources (`+`), and sinks (`-`) in real-time.
*   Press `T` to load the template maze, press `S` to dynamically scale the geometry, and change the view modes (`V`) to show density/direction.
*   This perfectly hits the "Entertain" requirement.

#### 5. Conclusion & Q&A (1-2 mins)
*   Summary of achievements (Validation, Scaling, Interactivity).
*   Open the floor for the 15–20 minutes of questions.

---

### 🎨 Graphics Generation (LaTeX / TikZ)

You can compile these standalone LaTeX documents to generate crisp, vector-graphics PDFs for your presentation slides.

#### Graphic 1: The D2Q9 Lattice Model
This visual explains the 9 discrete velocities in your simulation.
```latex
\documentclass[tikz, border=10pt]{standalone}
\usepackage{pgfplots}
\begin{document}
\begin{tikzpicture}[scale=2, very thick]
    % Grid lines
    \draw[step=1cm,gray!30,very thin] (-1.5,-1.5) grid (1.5,1.5);
    
    % Center node
    \filldraw[black] (0,0) circle (3pt) node[anchor=north west] {$f_0$ (Rest)};
    
    % Axial directions (Weight 1/9)
    \draw[->, blue, line width=1.5pt] (0,0) -- (1,0) node[anchor=west] {$f_1$};
    \draw[->, blue, line width=1.5pt] (0,0) -- (0,1) node[anchor=south] {$f_2$};
    \draw[->, blue, line width=1.5pt] (0,0) -- (-1,0) node[anchor=east] {$f_3$};
    \draw[->, blue, line width=1.5pt] (0,0) -- (0,-1) node[anchor=north] {$f_4$};
    
    % Diagonal directions (Weight 1/36)
    \draw[->, red, line width=1.5pt] (0,0) -- (1,1) node[anchor=south west] {$f_5$};
    \draw[->, red, line width=1.5pt] (0,0) -- (-1,1) node[anchor=south east] {$f_6$};
    \draw[->, red, line width=1.5pt] (0,0) -- (-1,-1) node[anchor=north east] {$f_7$};
    \draw[->, red, line width=1.5pt] (0,0) -- (1,-1) node[anchor=north west] {$f_8$};
    
    % Annotations
    \node[blue, align=center] at (1.5, 0.2) {$w = 1/9$};
    \node[red, align=center] at (1.2, 1.2) {$w = 1/36$};
    \node[black] at (0.3, -0.3) {$w = 4/9$};
\end{tikzpicture}
\end{document}
```

#### Graphic 2: 2D Domain Decomposition & Halo Exchange
This visually explains your `halo_exchange_2d()` function (Milestone 06).
```latex
\documentclass[tikz, border=10pt]{standalone}
\usetikzlibrary{patterns}
\begin{document}
\begin{tikzpicture}[scale=1.5]
    % Rank 0 (Center)
    \fill[blue!10] (0,0) rectangle (2,2);
    \draw[thick, blue] (0,0) rectangle (2,2);
    \node[blue] at (1,1) {\Large Rank $P_{x,y}$};
    
    % Halo buffers for Rank 0
    \fill[pattern=north west lines, pattern color=blue!50] (0,2) rectangle (2,2.25); % North
    \fill[pattern=north west lines, pattern color=blue!50] (0,-0.25) rectangle (2,0); % South
    \fill[pattern=north west lines, pattern color=blue!50] (2,0) rectangle (2.25,2); % East
    \fill[pattern=north west lines, pattern color=blue!50] (-0.25,0) rectangle (0,2); % West
    
    % Corners
    \fill[red!40] (2,2) rectangle (2.25, 2.25); % NE
    \fill[red!40] (-0.25,2) rectangle (0, 2.25); % NW
    \fill[red!40] (2,-0.25) rectangle (2.25, 0); % SE
    \fill[red!40] (-0.25,-0.25) rectangle (0, 0); % SW

    % Neighbor Ranks (Partial views)
    \draw[dashed, thick, gray] (0,2.25) rectangle (2,3); \node[gray] at (1,2.6) {$P_{x, y+1}$ (North)};
    \draw[dashed, thick, gray] (0,-1) rectangle (2,-0.25); \node[gray] at (1,-0.6) {$P_{x, y-1}$ (South)};
    \draw[dashed, thick, gray] (2.25,0) rectangle (3,2); \node[gray, rotate=-90] at (2.6,1) {$P_{x+1, y}$ (East)};
    \draw[dashed, thick, gray] (-1,0) rectangle (-0.25,2); \node[gray, rotate=90] at (-0.6,1) {$P_{x-1, y}$ (West)};

    % Communication Arrows
    \draw[->, thick, black] (1, 1.8) -- (1, 2.4) node[midway, right] {Send $f_{2,5,6}$};
    \draw[<-, thick, black] (0.8, 1.8) -- (0.8, 2.4) node[midway, left] {Recv $f_{4,7,8}$};
\end{tikzpicture}
\end{document}
```

---

### 📈 Presentation-Ready Python Graphs

You provided the raw data for Milestone 6 in `milestone06/graph.py`. Here is an upgraded, presentation-styled script that creates a beautiful, dark-theme-friendly (or clean light-theme) dual-axis graph. It's designed to look professional on a projector.

```python
import matplotlib.pyplot as plt
import numpy as np

# Data from milestone06
ranks = np.array([1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125])
runtime_s = np.array([43.0904, 21.5256, 11.5014, 9.80591, 6.35969, 6.21572, 7.17903, 7.71452, 11.6942, 13.8975, 18.9978, 19.1994])
performance_mlups = np.array([23.207, 46.4564, 86.9456, 101.979, 157.24, 160.882, 139.295, 129.626, 85.5128, 71.9555, 52.6377, 52.085])

# Ideal scaling line (based on Rank 1)
ideal_runtime = runtime_s[0] / ranks

# Setup Plot styling for presentations
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

color1 = "#2c3e50" # Dark Blue
color2 = "#e74c3c" # Red
ideal_color = "#95a5a6" # Grey

# --- Left Axis (Runtime) ---
ax1.set_xlabel("Number of MPI Ranks (GPUs/Cores)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Runtime (s) $\downarrow$ Better", color=color1, fontsize=14, fontweight='bold')
l1 = ax1.plot(ranks, runtime_s, marker="o", markersize=8, linewidth=3, color=color1, label="Actual Runtime")
l_ideal = ax1.plot(ranks, ideal_runtime, linestyle="--", linewidth=2, color=ideal_color, label="Ideal Runtime")
ax1.tick_params(axis="y", labelcolor=color1, labelsize=12)
ax1.tick_params(axis="x", labelsize=12)

# --- Right Axis (Performance) ---
ax2 = ax1.twinx()
ax2.set_ylabel("Performance (MLUPS) $\uparrow$ Better", color=color2, fontsize=14, fontweight='bold')
l2 = ax2.plot(ranks, performance_mlups, marker="s", markersize=8, linewidth=3, color=color2, label="MLUPS")
ax2.tick_params(axis="y", labelcolor=color2, labelsize=12)

# Grid and Title
ax1.grid(True, linestyle=":", alpha=0.7)
plt.title("Strong Scaling Performance: 1000x1000 Grid (1000 Steps)", fontsize=16, fontweight='bold', pad=15)

# Highlight the "Sweet Spot" (where communication overtakes computation)
optimal_rank = ranks[np.argmax(performance_mlups)]
ax2.axvline(x=optimal_rank, color='goldenrod', linestyle='-.', alpha=0.8, label=f"Optimal Peak ({optimal_rank} ranks)")

# Unified Legend
lines = l1 + l_ideal + l2 + [ax2.get_lines()[-1]]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper center", fontsize=12, frameon=True, shadow=True, bbox_to_anchor=(0.5, -0.15), ncol=4)

plt.tight_layout()
plt.savefig("presentation_scaling.png", bbox_inches="tight")
print("Generated presentation_scaling.png")
```

### 💡 Pro-Tips for the Presentation
1. **Be upfront about the Scaling dip:** In your graph, performance drops after 10-20 ranks. Don't hide this! Explain that for a `1000x1000` grid, the chunk per rank becomes too small ($< 100 \times 100$), causing MPI boundary communication (Halo Exchange) to dominate compute time (Amdahl's law bottleneck). This shows deep understanding.
2. **Interactive Demo:** Mention that the interactive demo runs locally on SDL3, but relies on the *exact same* Kokkos parallel backend underneath, proving the "Write Once, Run Anywhere" philosophy of Kokkos.




Here is a comprehensive presentation schema designed for a 10–15 minute talk, balancing technical depth with visual evidence and an entertaining hook. I have also included LaTeX (TikZ) codes for architectural diagrams and a Python script for generating a polished scaling efficiency plot.

---

# 📊 Presentation Schema (10–15 minutes)

**Target Audience:** Peers and instructors evaluating your HPC milestones.
**Core Message:** Our code is a highly scalable, visually verified, and hardware-portable D2Q9 Lattice Boltzmann fluid solver.

### **Slide 1: Title & Hook (Entertain) [1 min]**
*   **Title:** *Fluids at Scale: High-Performance Lattice Boltzmann with Kokkos & MPI*
*   **Visual:** A looping GIF from your FFmpeg video export (`milestone05_video.mp4`) showing the fluid reacting to obstacles.
*   **Hook:** "Solving the Navier-Stokes equations is notoriously hard. Today, I'll show you how we sidestepped the heavy math using particle distributions, scaled it across a supercomputer, and even made an interactive fluid-painting canvas."

### **Slide 2: What Does the Code Do? [1.5 mins]**
*   **Core Physics:** Solves fluid dynamics using the Lattice Boltzmann Method (LBM).
*   **Model:** D2Q9 (2D grid, 9 discrete particle velocities).
*   **Capabilities:**
    *   *Hardware Portability:* Uses **Kokkos** to run on CPUs or GPUs without changing the core algorithms.
    *   *Distributed Computing:* Uses **MPI** for multi-node execution with 1D and 2D domain decomposition.
    *   *Interactive & Multimedia:* Features real-time rendering via SDL3 and direct FFmpeg integration for video exports.

### **Slide 3: The LBM Algorithm (How it works) [1.5 mins]**
*   **Visual:** *Use the TikZ D2Q9 Lattice graphic provided below.*
*   **Concept:** Instead of macroscopic pressure and velocity, we track particle distributions $f_i$.
*   **Two main steps (run via Kokkos `parallel_for`):**
    1.  **Collision:** Particles relax towards a local equilibrium state (controlled by viscosity parameter $\omega$).
    2.  **Streaming:** Particles move to neighboring lattice cells.

### **Slide 4: Verification 1 – Shear Wave Decay (M04) [2 mins]**
*   **Goal:** Convince the audience the physics engine is accurate.
*   **Visuals:** Place your generated `velocity_profile.png` and `viscosity.png` side-by-side.
*   **Talking Points:**
    *   We simulated a decaying sinusoidal velocity profile.
    *   As predicted by theory, the velocity decays exponentially over time.
    *   By measuring the decay rate, we calculated the numerical kinematic viscosity. Our results perfectly match the analytical curve $\nu = \frac{1}{3}(\frac{1}{\omega} - \frac{1}{2})$. The physics works!

### **Slide 5: Verification 2 – Lid-Driven Cavity (M05) [2 mins]**
*   **Goal:** Prove the code handles complex boundary conditions (Bounce-back walls).
*   **Visuals:** Show `lid_driven_cavity.png` (streamlines) and `centerline_profile.png`.
*   **Talking Points:**
    *   The classic CFD benchmark: A box where the top lid moves at a constant speed, driving a vortex inside.
    *   Our simulation dynamically converges to the steady-state (Reynolds number $\sim 435$).
    *   The streamline plot clearly shows the primary central vortex and secondary corner vortices.

### **Slide 6: Under the Hood – 2D Domain Decomposition [2 mins]**
*   **Goal:** Explain how you scaled the problem up.
*   **Visual:** *Use the TikZ Halo Exchange graphic provided below.*
*   **Talking Points:**
    *   To scale beyond one GPU, we divide the grid into a 2D Cartesian topology using MPI.
    *   Each node only computes its local domain.
    *   **Halo Exchange:** Before the streaming step, boundary cells ("ghost cells") are communicated between neighboring MPI ranks. We exchange faces (N, S, E, W) and corners.

### **Slide 7: Performance & Scaling (M06) [2.5 mins]**
*   **Goal:** Prove the HPC implementation is efficient.
*   **Visual:** *Use the Python Scaling Efficiency plot generated by the script below.* (Based on your `milestone06` data).
*   **Talking Points:**
    *   We benchmarked a fixed $1000 \times 1000$ grid over 1000 steps on bwUniCluster.
    *   Look at the runtime drop from 43s (1 rank) to ~6s (8-10 ranks).
    *   **Amdahl’s Law Bottleneck:** As we push past 10 ranks for this specific grid size, communication overhead (Halo exchanges) begins to outweigh computation. This proves our code is highly optimized, making the problem memory-bandwidth/network bound very quickly!

### **Slide 8: The Interactive Canvas (Entertain) [1.5 mins]**
*   **Goal:** End on a high, interactive note.
*   **Visual:** Live demo or a screen recording of the `./interactive` target.
*   **Talking Points:**
    *   Because our Kokkos implementation is so fast, we built an interactive wrapper.
    *   With SDL3, users can "paint" walls, fluid sources, and sinks in real-time.
    *   Demonstrate changing the fluid view dynamically (Speed / Direction / Density).

### **Slide 9: Q&A [1 min]**
*   "Thank you! We're ready to answer your questions regarding Kokkos performance, MPI synchronization, or LBM stability."

---

# 🎨 Graphics Generation

### 1. LaTeX (TikZ): D2Q9 Lattice Model
*Use this graphic on Slide 3. It cleanly illustrates the 9 discrete velocities.*

```latex
\documentclass{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=2, >=stealth, thick]
    % Grid lines
    \draw[lightgray, dashed] (-1.5,-1.5) grid (1.5,1.5);
    
    % Vectors
    \draw[->, blue] (0,0) -- (1,0) node[anchor=west] {$c_1$};
    \draw[->, blue] (0,0) -- (0,1) node[anchor=south] {$c_2$};
    \draw[->, blue] (0,0) -- (-1,0) node[anchor=east] {$c_3$};
    \draw[->, blue] (0,0) -- (0,-1) node[anchor=north] {$c_4$};
    
    \draw[->, red] (0,0) -- (1,1) node[anchor=south west] {$c_5$};
    \draw[->, red] (0,0) -- (-1,1) node[anchor=south east] {$c_6$};
    \draw[->, red] (0,0) -- (-1,-1) node[anchor=north east] {$c_7$};
    \draw[->, red] (0,0) -- (1,-1) node[anchor=north west] {$c_8$};
    
    % Center node
    \filldraw[black] (0,0) circle (2pt) node[anchor=south east] {$c_0$};
    
    % Weights text
    \node[align=center, fill=white, inner sep=2pt] at (0,-1.7) {
        $w_0 = 4/9$ \\
        $w_{1-4} = 1/9$ \\
        $w_{5-8} = 1/36$
    };
\end{tikzpicture}
\end{document}
```

### 2. LaTeX (TikZ): 2D Domain Decomposition & Halo Exchange
*Use this graphic on Slide 6 to visually explain MPI ghost cells (halos).*

```latex
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{patterns}
\begin{document}
\begin{tikzpicture}[scale=0.8, thick]
    
    % Core domains (4 MPI ranks)
    \fill[blue!10] (0.5,0.5) rectangle (4.5,4.5);
    \fill[green!10] (5.5,0.5) rectangle (9.5,4.5);
    \fill[orange!10] (0.5,5.5) rectangle (4.5,9.5);
    \fill[purple!10] (5.5,5.5) rectangle (9.5,9.5);
    
    % Draw grids for the bottom-left rank to show cells
    \draw[step=1.0, blue!30, thin] (0.5,0.5) grid (4.5,4.5);
    \draw[step=1.0, green!30, thin] (5.5,0.5) grid (9.5,4.5);
    \draw[step=1.0, orange!30, thin] (0.5,5.5) grid (4.5,9.5);
    
    \node at (2.5,2.5) {\Large \textbf{Rank 0}};
    \node at (7.5,2.5) {\Large \textbf{Rank 1}};
    \node at (2.5,7.5) {\Large \textbf{Rank 2}};
    \node at (7.5,7.5) {\Large \textbf{Rank 3}};

    % Ghost cells (Halos) for Rank 0
    \draw[dashed, red, very thick] (0.5, 4.5) rectangle (4.5, 5.5); % Top Halo
    \draw[dashed, red, very thick] (4.5, 0.5) rectangle (5.5, 4.5); % Right Halo
    \draw[dashed, red, very thick] (4.5, 4.5) rectangle (5.5, 5.5); % Corner Halo

    % Exchange arrows
    \draw[->, ultra thick, black] (2.5, 5) -- (2.5, 6);
    \draw[<-, ultra thick, black, dashed] (3.5, 5) -- (3.5, 6);
    
    \draw[->, ultra thick, black] (5, 2.5) -- (6, 2.5);
    \draw[<-, ultra thick, black, dashed] (5, 1.5) -- (6, 1.5);

    % Legend
    \draw[dashed, red, very thick] (11, 8.5) rectangle (12, 9.5);
    \node[right] at (12.2, 9) {Ghost Cells (Halo)};
    
    \draw[->, ultra thick, black] (11, 7.5) -- (12, 7.5);
    \node[right] at (12.2, 7.5) {MPI Send};
    
    \draw[<-, ultra thick, black, dashed] (11, 6.5) -- (12, 6.5);
    \node[right] at (12.2, 6.5) {MPI Recv};
    
\end{tikzpicture}
\end{document}
```

### 3. Python: Enhanced Scaling Plot (Strong Scaling Efficiency)
*Your `milestone06/graph.py` shows absolute runtime. In HPC presentations, an **Ideal vs. Actual Speedup** plot is much more impressive and professional for Slide 7.*

```python
import matplotlib.pyplot as plt
import numpy as np

# Data from Milestone 06
ranks = np.array([1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125])
runtime_s = np.array([43.0904, 21.5256, 11.5014, 9.80591, 6.35969, 6.21572, 7.17903, 7.71452, 11.6942, 13.8975, 18.9978, 19.1994])

# Calculate Speedup
# Speedup = Runtime(1 rank) / Runtime(N ranks)
base_time = runtime_s[0]
speedup = base_time / runtime_s

# Setup plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Actual Speedup
ax1.plot(ranks, speedup, marker='o', markersize=8, linewidth=2.5, color='#d62728', label='Actual Speedup')

# Plot Ideal Speedup (Linear)
# We limit ideal line to 20 since actual performance degrades after 10-20 due to comms
ax1.plot(ranks[ranks <= 20], ranks[ranks <= 20], linestyle='--', color='gray', linewidth=2, label='Ideal Speedup')

# Highlight the "Sweet Spot" (max efficiency)
max_idx = np.argmax(speedup)
ax1.annotate(f'Peak Efficiency\n{ranks[max_idx]} Ranks', 
             xy=(ranks[max_idx], speedup[max_idx]), 
             xytext=(ranks[max_idx]+10, speedup[max_idx]+1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold')

# Formatting
ax1.set_title("Strong Scaling Speedup: 1000x1000 Grid (1000 steps)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("Number of MPI Ranks", fontsize=13)
ax1.set_ylabel("Speedup factor", fontsize=13)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.set_xlim(0, 130)
ax1.set_ylim(0, 8)
ax1.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)

# Add text explaining the drop
textstr = "Communication overhead\n(Halo exchanges) dominates\ncomputation past 20 ranks."
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(70, 2, textstr, fontsize=11, verticalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig("scaling_efficiency.png", dpi=300)
print("Saved scaling_efficiency.png")
```
"""
Lorenz Attractor Simulation
============================
Integrates the Lorenz system, plots the phase-space trajectory, and
exports two animation formats so it can be embedded in a Beamer slide:

  1. A sequence of PNG frames  -> use with the `animate` package
     (\\animategraphics{...})
  2. An MPEG video file        -> use with the `multimedia` package
     or a plain hyperlink / media player embed

See files for Sep 4 on Canvas.

Outputs (all written to ./lorenz_output/):
  lorenz_static.png        - single high-res phase-space plot
  frames/frame0000.png ...  - PNG frame sequence
  lorenz_animation.mpg      - MPEG-1 video
  lorenz_animation.mp4      - MP4 video (bonus, widely compatible)
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")  # no display needed, just render to file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import matplotlib.animation as animation

# ----------------------------------------------------------------------
# 1. Lorenz system parameters (classic chaotic regime)
# ----------------------------------------------------------------------
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

def lorenz(t, state, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# ----------------------------------------------------------------------
# 2. Integrate the ODE
# ----------------------------------------------------------------------
t_start, t_end = 0.0, 40.0
n_points = 4000
t_eval = np.linspace(t_start, t_end, n_points)
initial_state = [1.0, 1.0, 1.0]

sol = solve_ivp(
    lorenz, [t_start, t_end], initial_state,
    t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8
)
x, y, z = sol.y

# ----------------------------------------------------------------------
# 3. Output directories
# ----------------------------------------------------------------------
OUT_DIR = "lorenz_output"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 4. Static phase-space plot (full trajectory)
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, lw=0.6, color="royalblue")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor — Phase Space")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lorenz_static.png"), dpi=200)
plt.close(fig)
print("Saved static phase-space plot.")

# ----------------------------------------------------------------------
# 5. Animation setup (growing trajectory + moving point)
# ----------------------------------------------------------------------
N_FRAMES = 20                     # total frames in the animation
stride = n_points // N_FRAMES      # sample the solution evenly

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor")

line, = ax.plot([], [], [], lw=0.8, color="royalblue")
point, = ax.plot([], [], [], "o", color="crimson", markersize=4)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def update(frame_idx):
    i = min(frame_idx * stride, n_points - 1)
    line.set_data(x[:i], y[:i])
    line.set_3d_properties(z[:i])
    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])
    ax.view_init(elev=25, azim=0.5 * frame_idx)  # slow rotation
    return line, point

anim = animation.FuncAnimation(
    fig, update, frames=N_FRAMES, init_func=init, blit=False
)

# ----------------------------------------------------------------------
# 6a. Export PNG frame sequence (for LaTeX \animategraphics)
# ----------------------------------------------------------------------
for i in range(N_FRAMES):
    update(i)
    fig.savefig(os.path.join(FRAMES_DIR, f"frame{i:04d}.png"), dpi=120)
print(f"Saved {N_FRAMES} PNG frames to {FRAMES_DIR}/")

# ----------------------------------------------------------------------
# 6b. Export MPEG video (and MP4 as a bonus, widely-supported format)
# ----------------------------------------------------------------------
writer_mpg = animation.FFMpegWriter(fps=20, codec="mpeg1video", bitrate=4000)
anim.save(os.path.join(OUT_DIR, "lorenz_animation.mpg"), writer=writer_mpg)
print("Saved MPEG animation.")

writer_mp4 = animation.FFMpegWriter(fps=20, bitrate=4000)
anim.save(os.path.join(OUT_DIR, "lorenz_animation.mp4"), writer=writer_mp4)
print("Saved MP4 animation.")

plt.close(fig)
print("Done. All outputs are in:", os.path.abspath(OUT_DIR))

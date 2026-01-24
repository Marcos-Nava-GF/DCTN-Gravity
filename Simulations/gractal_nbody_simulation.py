import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pandas as pd
import os

# Parameters from DCTN Theory
E_i = 67.88           # Base expansion (Planck/Early)
alpha_base = 0.025    # Delta correction (+2.5%)
d_H = 1.41            # Hausdorff Dimension (Gractal)
N_nodes = 2000        # Simulation scale

# 1. Generate a Filamentous Structure (simplified Gractal skeleton)
def generate_gractal_skeleton(n_points, d_h):
    np.random.seed(42)  # Reproducibility
    points = np.zeros((n_points, 3))
    for i in range(1, n_points):
        step = np.random.normal(0, 1, 3)
        step /= np.linalg.norm(step)
        # Using the fractal scaling logic: step size scales with index to affect dimension
        # A negative exponent (1/dH - 1 < 0) implies decreasing steps, creating clusters?
        # Actually standard fractional brownian motion uses H = 1/dH. 
        # Here we use the user's provided heuristic.
        points[i] = points[i-1] + step * (i**(1/d_h - 1))
    return points

points = generate_gractal_skeleton(N_nodes, d_H)

# 2. Calculate Local Density (rho)
tree = KDTree(points)
radius = 5.0 # Radius in arbitrary simulation units
counts = tree.query_ball_point(points, r=radius, return_length=True)
# Density = count / volume
densities = np.array(counts) / (4/3 * np.pi * radius**3)
# Normalize relative to mean density (rho_bar = 1)
rho_rel = densities / np.mean(densities)

# 3. Apply the "Napkin" Equation: Ea = Ei + alpha_a(rho)
# We assume alpha_a is higher in low density regions (Voids)
def alpha_a_func(rho, alpha):
    # Feedback loop: Efficiency ~ 1 / sqrt(rho)
    # Avoid division by zero
    rho_safe = np.maximum(rho, 0.01) 
    return alpha * (1.0 / (rho_safe**0.5))

# Calculate local expansion
alpha_a_vals = alpha_a_func(rho_rel, alpha_base)
E_a_local = E_i * (1 + alpha_a_vals)

# 4. Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Color nodes by local Expansion Rate Ea
# Voids (High Ea) -> Red/Warm, Hubs (Low Ea) -> Blue/Cool
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c=E_a_local, cmap='plasma', s=20, alpha=0.8)

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Local Expansion Rate $E_a$ (km/s/Mpc)')

ax.set_title(f"Gractal N-Body Simulation ($d_H \\approx {d_H}$)\n$E_a = E_i + \\alpha_a(\\rho)$", fontsize=14)
ax.set_xlabel("X (Mpc)")
ax.set_ylabel("Y (Mpc)")
ax.set_zlabel("Z (Mpc)")
ax.view_init(elev=20, azim=45)

# Save Plot
output_path_img = os.path.join("../../Preprints/images", 'p2_fig3_nbody_expansion.png')
os.makedirs(os.path.dirname(output_path_img), exist_ok=True)
plt.savefig(output_path_img, dpi=300)
print(f"Plot saved to {output_path_img}")

# Save CSV Data for verification
df_results = pd.DataFrame({
    'x': points[:, 0],
    'y': points[:, 1],
    'z': points[:, 2],
    'density_rel': rho_rel,
    'E_a_local': E_a_local
})
output_path_csv = 'gractal_simulation_data.csv'
df_results.to_csv(output_path_csv, index=False)
print(f"Data saved to {output_path_csv}")

# Statistics
stats = {
    'Mean Ea': np.mean(E_a_local),
    'Max Ea (Voids)': np.max(E_a_local),
    'Min Ea (Hubs)': np.min(E_a_local),
    'Target SHOES': 75.26,
    'Target Planck': 67.88
}
print("\n--- SIMULATION RESULTS ---")
for k, v in stats.items():
    print(f"{k}: {v:.2f}")

import numpy as np
import matplotlib.pyplot as plt
import os

# Base Parameters from DCTN Theory
# Base Parameters from DCTN Theory
E_i = 67.88          # Initial Expansion (Planck)
delta_h_base = 0.025   # The "+2.5%" Sweet Spot
d_H = 1.4151         # Hausdorff Dimension (Fractal)

# Local density simulation (0.1 = Voids, 5.0 = Galaxy Clusters/Hubs)
# rho = 1.0 is the mean density of the toy model
densities = np.linspace(0.1, 5.0, 100) 

def calculate_local_expansion(rho, base, delta_h, dim):
    """
    Calculates Ea based on the 'Density Scaling Law': Ea = Ei + delta_h
    Where delta_h depends on density rho and topology d_H.
    
    The dissipation efficiency decreases as the network saturates (Hubs).
    We use an inverse relation based on filamentous structure:
    Scaling factor ~ 1 / (rho ^ (2 - d_H))
    
    Why (2 - d_H)? 
    In a 2D sheet (holographic bound), scaling is linear. 
    In fractals, efficiency scales with the roughness.
    """
    # Inverse relation: Lower density -> Higher efficiency (Void effect)
    # Higher density -> Lower efficiency (Saturation effect)
    delta_local = delta_h * (1 / (rho**(2 - dim))) 
    
    # We apply the boost as a percentage of the base
    # Formula: Ea = Ei * (1 + delta_local)
    # Or additive: Ea = Ei + (Ei * delta_local)
    return base + (base * delta_local)

# Run simulation
E_a_values = [calculate_local_expansion(r, E_i, delta_h_base, d_H) for r in densities]

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(densities, E_a_values, label=r'Gractal Expansion $E_a(\rho)$', color='red', lw=2)

# Reference Lines
plt.axhline(y=75.26, color='blue', linestyle='--', label='SH0ES Target (Local/Voids)') 
plt.axhline(y=70.33, color='green', linestyle=':', label='DCTN Mean (Preprint 2)')
plt.axhline(y=E_i, color='gray', linestyle='-.', label='Planck Base (High Density)')

plt.title(f"Variation of Actual Expansion ($E_a$) vs Local Density ($d_H={d_H}$)")
plt.xlabel(r"Local Information Density ($\rho$)")
plt.ylabel(r"Expansion Rate ($km/s/Mpc$)")
plt.legend()
plt.grid(alpha=0.3)

# Save Plot
output_path = os.path.join("../../Preprints/images", 'p2_fig2_expansion_vs_density.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# Output Analysis
print(f"\n--- ANALYSIS RESULTS ---")
print(f"Expansion in Voids (rho=0.1): {E_a_values[0]:.2f} km/s/Mpc")
print(f"Expansion in Clusters (rho=5.0): {E_a_values[-1]:.2f} km/s/Mpc")

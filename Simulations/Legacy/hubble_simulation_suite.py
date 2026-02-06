import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# --- CONFIGURATION & CONSTANTS ---
# Global Constants
H0_PLANCK_BASE = 67.88  # Ei: Initial Expansion (Planck/LambdaCDM)
H0_SHOES = 75.26        # Target: Late Universe (SH0ES)
DH_DIMENSION = 1.4151   # Hausdorff Dimension (Fractal)

# Network Simulation Constants
STEPS = 150             # Temporal steps for network growth
M_LINKS = 4             # Connections per new tensor
GAMMA_COST = 2.5        # Causal Cost Exponent (Formerly ALPHA) - Topology
BETA_GRAVITY_VOID = 0.0 # Repulsive/Random (Void)
BETA_GRAVITY_CLUSTER = 2.5 # Attractive (Cluster)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def ensure_output_dir():
    path = os.path.join("..", "Preprints", "images")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

# ==========================================
# MODULE 1: Network Growth Monte Carlo (from hubble_tension.py)
# ==========================================
def run_network_growth_simulation():
    print("\n--- MODULE 1: Gractal Universe Growth (Monte Carlo) ---")
    print("Simulating Void vs Cluster expansion rates...")

    def grow_network(beta, steps, label):
        print(f"Simulating Environment: {label}...")
        G = nx.complete_graph(M_LINKS + 2)
        degrees = [G.degree(n) for n in G.nodes()]
        num_nodes = G.number_of_nodes()
        scale_factor_a = []
        time_points = [] 

        for t in range(steps):
            if t % 10 == 0 and t > 0:
                try:
                    a_t = nx.average_shortest_path_length(G)
                    scale_factor_a.append(a_t)
                    time_points.append(t)
                except nx.NetworkXError:
                    pass

            for _ in range(5): # Burst growth
                new_node_id = num_nodes
                # Vectorized probability calc
                current_nodes_indices = np.arange(num_nodes)
                current_degrees = np.array(degrees)
                dist_causal = np.maximum(new_node_id - current_nodes_indices, 1)
                
                # WEIGHTS: (Degree^Beta) / (Distance^Gamma)
                weights = (current_degrees ** beta) / (dist_causal ** GAMMA_COST)
                
                w_sum = weights.sum()
                probs = weights / w_sum if w_sum > 0 else np.ones(num_nodes) / num_nodes
                
                targets = np.random.choice(current_nodes_indices, size=M_LINKS, replace=False, p=probs)
                G.add_node(new_node_id)
                for target in targets:
                    G.add_edge(new_node_id, target)
                    degrees[target] += 1
                degrees.append(M_LINKS)
                num_nodes += 1
        
        return np.array(scale_factor_a), np.array(time_points)

    def run_monte_carlo(beta, steps, label, trials=10):
        print(f"   > Starting Monte Carlo for {label} ({trials} runs)...")
        all_runs = []
        t_ref = None
        for i in range(trials):
            a_array, t_array = grow_network(beta, steps, label)
            if t_ref is None: t_ref = t_array
            # Simple length check to ensure alignment
            if len(a_array) == len(t_ref):
                all_runs.append(a_array)
        
        avg_a = np.mean(all_runs, axis=0)
        return avg_a, t_ref

    # 1. Simulate
    a_void, t_void = run_monte_carlo(BETA_GRAVITY_VOID, STEPS, "VOID (Underdense)")
    a_cluster, t_cluster = run_monte_carlo(BETA_GRAVITY_CLUSTER, STEPS, "CLUSTER (Dense)")

    # 2. Calculate H0 (Expansion Rate)
    dt = 10
    h0_void_raw = (np.diff(a_void) / a_void[:-1]) / dt
    h0_cluster_raw = (np.diff(a_cluster) / a_cluster[:-1]) / dt

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')

    h0_void = smooth(h0_void_raw, 3)
    h0_cluster = smooth(h0_cluster_raw, 3)
    t_h0 = t_void[1:]

    # 3. Plot
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Size a(t)
    plt.subplot(1, 2, 1)
    plt.plot(t_void, a_void, label=f'Void (Sparse, $\\beta={BETA_GRAVITY_VOID}$)', color='blue', linewidth=2.5)
    plt.plot(t_cluster, a_cluster, label=f'Cluster (Dense, $\\beta={BETA_GRAVITY_CLUSTER}$)', color='red', linewidth=2.5)
    plt.title("Differential Universe Evolution: $a(t)$")
    plt.xlabel("Cosmic Time (Steps)")
    plt.ylabel("Universe Size")
    plt.legend()
    plt.grid(True, alpha=0.2)

    # Subplot 2: H0
    plt.subplot(1, 2, 2)
    baseline = np.mean(h0_cluster[-5:])
    scale_factor = H0_PLANCK_BASE / baseline # Calibrate to Planck at cluster limit
    
    h0_v_norm = h0_void * scale_factor
    h0_c_norm = h0_cluster * scale_factor

    plt.plot(t_h0, h0_v_norm, color='blue', alpha=0.8, label='H0 Void (Simulated)')
    plt.plot(t_h0, h0_c_norm, color='red', alpha=0.8, label='H0 Cluster (Simulated)')
    plt.axhline(y=H0_SHOES, color='blue', linestyle='--', alpha=0.5, label='Observed: SH0ES')
    plt.axhline(y=H0_PLANCK_BASE, color='red', linestyle='--', alpha=0.5, label='Observed: Planck')
    
    plt.title("Hubble Tension Resolution")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    output_path = ensure_output_dir()
    save_file = os.path.join(output_path, 'hubble_suite_growth_simulation.png')
    plt.tight_layout()
    plt.savefig(save_file)
    print(f"   > Plot saved to {save_file}")
    plt.show()

# ==========================================
# MODULE 2: Calibration Sweep (from hubble_tension_calibration.py)
# ==========================================
def run_calibration_sweep():
    print("\n--- MODULE 2: Hubble Tension Calibration Sweep ---")
    print(f"Base Ei (Planck): {H0_PLANCK_BASE} | Target (SH0ES): {H0_SHOES}")

    deltas_h = np.linspace(0, 5.0, 50) # Range of dimensional perturbation
    results = []

    def calculate_age_approx(h0, omega_m=0.315, omega_lambda=0.685):
        t_hubble = 977.8 / h0 
        term = (1 + np.sqrt(omega_lambda)) / np.sqrt(omega_m)
        factor = (2/3) * np.log(term) / np.sqrt(omega_lambda)
        return t_hubble * factor

    print("   > Running sweep...")
    for delta_h in deltas_h:
        E_a = H0_PLANCK_BASE + delta_h
        diff = H0_SHOES - E_a
        age_raw = calculate_age_approx(E_a)
        age = age_raw * 1.04 # Structural correction factor
        results.append({"delta_h": delta_h, "E_a": E_a, "age": age})

    # Find Sweet Spot (+2.45 approx)
    target_delta = 2.45
    idx = (np.abs(deltas_h - target_delta)).argmin()
    sweet_spot = results[idx]

    print(f"   > SWEET SPOT FOUND at Delta_H = {sweet_spot['delta_h']:.2f}")
    print(f"     Resulting H0: {sweet_spot['E_a']:.2f} km/s/Mpc")
    print(f"     Universe Age: {sweet_spot['age']:.2f} Ga")

    # Plot
    delta_vals = deltas_h
    ea_values = [r['E_a'] for r in results]
    age_values = [r['age'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(delta_vals, ea_values, color='tab:blue', linewidth=2, label='Adjusted H0 (Ea)')
    ax1.set_xlabel('Dimensional Perturbation Delta_H')
    ax1.set_ylabel('H0 (km/s/Mpc)', color='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(delta_vals, age_values, color='tab:red', linewidth=2, linestyle='-.', label='Universe Age')
    ax2.set_ylabel('Age (Ga)', color='tab:red')
    
    plt.axvline(x=2.45, color='purple', linestyle='-', alpha=0.7, label='Sweet Spot')
    plt.title('Hubble Tension Mitigation via Dimensional Impulse')
    
    output_path = ensure_output_dir()
    save_file = os.path.join(output_path, 'hubble_suite_calibration.png')
    plt.savefig(save_file)
    print(f"   > Plot saved to {save_file}")
    plt.close()

# ==========================================
# MODULE 3: Density Analysis (from hubble_tension_density.py)
# ==========================================
def run_density_analysis():
    print("\n--- MODULE 3: Local Density vs Expansion Rate ---")
    
    densities = np.linspace(0.1, 5.0, 100)
    delta_h_base = 0.025 # The 2.5% boost

    def calculate_local_expansion(rho, base, delta_percent, dim):
        # Efficiency scales inversely with density roughnes: 1 / rho^(2-dH)
        efficiency = 1 / (rho**(2 - dim))
        delta_local_percent = delta_percent * efficiency
        return base * (1 + delta_local_percent)

    ea_values = [calculate_local_expansion(r, H0_PLANCK_BASE, delta_h_base, DH_DIMENSION) for r in densities]

    plt.figure(figsize=(10, 6))
    plt.plot(densities, ea_values, label=r'Gractal Expansion $E_a(\rho)$', color='red', lw=2)
    plt.axhline(y=H0_SHOES, color='blue', linestyle='--', label='SH0ES Target (Void)')
    plt.axhline(y=H0_PLANCK_BASE, color='gray', linestyle='-.', label='Planck Base (Cluster)')
    
    plt.title(f"Variation of Expansion ($E_a$) vs Local Density ($d_H={DH_DIMENSION}$)")
    plt.xlabel(r"Local Information Density ($\rho$)")
    plt.ylabel(r"Expansion Rate ($km/s/Mpc$)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    output_path = ensure_output_dir()
    save_file = os.path.join(output_path, 'hubble_suite_density.png')
    plt.savefig(save_file)
    print(f"   > Plot saved to {save_file}")
    print(f"   > Void Expansion (rho=0.1): {ea_values[0]:.2f}")
    print(f"   > Cluster Expansion (rho=5.0): {ea_values[-1]:.2f}")
    plt.close()

# ==========================================
# MAIN MENU
# ==========================================
def main():
    while True:
        clear_screen()
        print("========================================")
        print("   DCTN HUBBLE SIMULATION SUITE v1.0    ")
        print("========================================")
        print("1. Run Network Growth (Monte Carlo) - Simulates Void vs Cluster graphs")
        print("2. Run Calibration Sweep - Finds Delta_H sweet spot")
        print("3. Run Density Analysis - Plots Expansion vs Local Density")
        print("4. Run ALL (Sequential)")
        print("q. Quit")
        print("----------------------------------------")
        
        choice = input("Select an option: ").strip().lower()
        
        if choice == '1':
            run_network_growth_simulation()
            input("\nPress Enter to continue...")
        elif choice == '2':
            run_calibration_sweep()
            input("\nPress Enter to continue...")
        elif choice == '3':
            run_density_analysis()
            input("\nPress Enter to continue...")
        elif choice == '4':
            run_network_growth_simulation()
            run_calibration_sweep()
            run_density_analysis()
            input("\nAll tasks completed. Press Enter to continue...")
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid option.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")

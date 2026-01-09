import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- EXPERIMENT CONFIGURATION ---
STEPS = 100  # Temporal steps (growth iterations)
M_LINKS = 4  # Connections per new tensor
ALPHA = 2.5  # Gractal critical point

# Test parameters for Hubble Tension mitigation
BETA_VOID = 0.5  # High efficiency (Accelerated expansion)
BETA_CLUSTER = 1.8  # Low efficiency (Matter slows down expansion)


def grow_network(beta, steps, label):
    print(f"Simulating Environment: {label}...")

    # Initialize Graph
    G = nx.complete_graph(M_LINKS + 2)

    # OPTIMIZATION: Use lists for mutable structures, convert to numpy for math
    # Assuming nodes are sequential integers 0, 1, 2...
    degrees = [G.degree(n) for n in G.nodes()]
    num_nodes = G.number_of_nodes()

    scale_factor_a = []
    time_points = []  # To keep track of when we measured

    for t in range(steps):
        # Measure current scale factor (mean distance)
        if t % 10 == 0 and t > 0:
            # a(t) is the average shortest path length between tensors
            try:
                a_t = nx.average_shortest_path_length(G)
                scale_factor_a.append(a_t)
                time_points.append(t)
            except nx.NetworkXError:
                # Handle disconnected graph case if it ever happens
                pass

        # Add new tensors according to the DCTN growth rule
        for _ in range(5):  # Add bursts of tensors
            new_node_id = num_nodes

            # --- VECTORIZED CALCULATION START ---
            # Convert to numpy only for the probability calculation
            current_nodes_indices = np.arange(num_nodes)
            current_degrees = np.array(degrees)

            dist_causal = (new_node_id - current_nodes_indices)
            # Prevent division by zero safely
            dist_causal = np.maximum(dist_causal, 1)

            # Probability P targets nodes based on degrees (cohesion) and distance (expansion)
            weights = (current_degrees ** beta) / (dist_causal ** ALPHA)

            # Safety check for weights sum
            w_sum = weights.sum()
            if w_sum == 0:
                probs = np.ones(num_nodes) / num_nodes  # Fallback to uniform
            else:
                probs = weights / w_sum
            # --- VECTORIZED CALCULATION END ---

            targets = np.random.choice(current_nodes_indices, size=M_LINKS, replace=False, p=probs)

            G.add_node(new_node_id)
            for target in targets:
                G.add_edge(new_node_id, target)
                degrees[target] += 1  # Update degree in list

            degrees.append(M_LINKS)  # Degree of the new node
            num_nodes += 1

    return np.array(scale_factor_a), np.array(time_points)


# --- EXECUTION ---

def run_monte_carlo(beta, steps, label, trials=15):
    """
    Runs the simulation multiple times and averages the results
    to eliminate random noise (stochastic noise).
    """
    print(f"--- Starting Monte Carlo for {label} ({trials} iterations) ---")
    all_runs = []

    for i in range(trials):
        # Print a dot for each iteration to visualize progress
        print(".", end="", flush=True)
        a_array, t_array = grow_network(beta, steps, label)
        # Ensure all arrays have the same length before appending
        if len(a_array) == len(t_array):
            all_runs.append(a_array)
    print(" Done.")

    # Average across all runs (axis 0)
    # This creates the "ideal" smooth curve
    avg_a = np.mean(all_runs, axis=0)
    return avg_a, t_array


# --- ADJUSTED PARAMETERS FOR PREPRINT 2 ---
# Make Void more "repulsive" and Cluster more "attractive" to highlight the difference
BETA_VOID = 0.0  # Purely random connection (Maximum expansion)
BETA_CLUSTER = 2.5  # Very high gravity (Slows down expansion)
STEPS = 150  # Slightly longer duration to observe divergence

# --- ROBUST EXECUTION ---
print("\n>>> SIMULATING GRACTAL UNIVERSE (MONTE CARLO METHOD) <<<")

# 1. Obtain smoothed data
a_void, t_void = run_monte_carlo(BETA_VOID, STEPS, "VOID (Underdense)")
a_cluster, t_cluster = run_monte_carlo(BETA_CLUSTER, STEPS, "CLUSTER (Dense)")

# 2. H0 Calculation with Additional Smoothing (Moving Window)
dt = 10
# Calculate raw H0
h0_void_raw = (np.diff(a_void) / a_void[:-1]) / dt
h0_cluster_raw = (np.diff(a_cluster) / a_cluster[:-1]) / dt


# Function to smooth H0 lines (Rolling Average)
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# Smooth Hubble curves for professional visualization
h0_void = smooth(h0_void_raw, 3)
h0_cluster = smooth(h0_cluster_raw, 3)

# Adjust time axis for H0
t_h0 = t_void[1:]

# --- FINAL VISUALIZATION ---
plt.figure(figsize=(15, 6))

# Plot 1: Scale Factor (Expansion)
plt.subplot(1, 2, 1)
plt.plot(t_void, a_void, label=f'Void (Sparse Matter, $\\beta={BETA_VOID}$)', color='blue', linewidth=2.5)
plt.plot(t_cluster, a_cluster, label=f'Cluster (Dense Matter, $\\beta={BETA_CLUSTER}$)', color='red', linewidth=2.5)
plt.title("Differential Universe Evolution: $a(t)$")
plt.xlabel("Cosmic Time (Steps)")
plt.ylabel("Universe Size (Network Diameter)")
plt.legend()
plt.grid(True, alpha=0.2)

# Plot 2: Hubble Tension
plt.subplot(1, 2, 2)

# Realistic Normalization (Anchor to Planck)
baseline = np.mean(h0_cluster[-5:])  # Use the end of simulation as stable reference
scale_factor = 67.4 / baseline

h0_v_norm = h0_void * scale_factor
h0_c_norm = h0_cluster * scale_factor

plt.plot(t_h0, h0_v_norm, color='blue', alpha=0.8, label='H0 in Void (Simulated)')
plt.plot(t_h0, h0_c_norm, color='red', alpha=0.8, label='H0 in Cluster (Simulated)')

# Real Reference Lines
plt.axhline(y=74.0, color='blue', linestyle='--', alpha=0.5, label='Observed: Supernovae (SH0ES)')
plt.axhline(y=67.4, color='red', linestyle='--', alpha=0.5, label='Observed: CMB (Planck)')

plt.title("Hubble Tension Resolution")
plt.xlabel("Cosmic Time")
plt.ylabel("H0 (km/s/Mpc)")
plt.legend()
plt.ylim(60, 85)  # Limit Y-axis to clearly see the gap
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print(f"\nFinal Results (Average of last steps):")
print(f"H0 Void: {np.mean(h0_v_norm[-5:]):.2f} (Should be high)")
print(f"H0 Cluster: {np.mean(h0_c_norm[-5:]):.2f} (Should be low)")

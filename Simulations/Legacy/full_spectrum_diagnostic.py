import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings

def diagnose_universe_spectrum(G):
    print("\n🩺 FULL SPECTRUM DIAGNOSTIC (Unfiltered)...")

    # 1. Candidate Selection (More relaxed)
    degrees = np.array([d for n, d in G.degree()])
    avg_deg = np.mean(degrees)

    # We look from near-normal nodes (1.2x) up to medium hubs (20x)
    candidates = [
        n for n in G.nodes()
        if G.degree(n) > 1.2 * avg_deg and G.degree(n) < 20.0 * avg_deg
    ]

    # Random sampling of 1000 candidates for speed
    if len(candidates) > 1000:
        sample = np.random.choice(candidates, 1000, replace=False)
    else:
        sample = candidates

    print(f"   -> Analyzing raw sample of {len(sample)} nodes...")

    masses = []
    alphas = []
    ids = []

    for node in sample:
        # Radius 1 (Core)
        subgraph = nx.ego_graph(G, node, radius=1)
        n_core = len(subgraph)

        # Minimal sanity filter (must have at least one triangle)
        if n_core < 3:
            continue

        # Alpha calculation
        boundary = 0
        for n in subgraph.nodes():
            for nbr in G.neighbors(n):
                if nbr not in subgraph:
                    boundary += 1

        if boundary == 0:
            continue

        # Parameters (standard values)
        ds = 1.25
        dH = 1.41
        shielding = dH / ds
        alpha = boundary / (n_core ** shielding)

        masses.append(n_core)
        alphas.append(alpha)
        ids.append(node)

    # --- DATA VISUALIZATION ---
    if not masses:
        print("Critical error: The network is disconnected or empty.")
        return

    # Basic Statistics
    print(f"\n📊 ZOO STATISTICS (N={len(masses)})")
    print(f"   Mass (Nodes): Min={min(masses)}, Max={max(masses)}, Mean={np.mean(masses):.1f}")
    print(f"   Alpha: Min={min(alphas):.6f}, Max={max(alphas):.6f}, Mean={np.mean(alphas):.6f}")

    # Search for the densest cluster (Where the 'Real Particle' is)
    # We use a histogram to find the mode
    hist_alpha, bins_alpha = np.histogram(alphas, bins=30)
    peak_alpha_idx = np.argmax(hist_alpha)
    mode_alpha = (bins_alpha[peak_alpha_idx] + bins_alpha[peak_alpha_idx + 1]) / 2

    hist_mass, bins_mass = np.histogram(masses, bins=30)
    peak_mass_idx = np.argmax(hist_mass)
    mode_mass = (bins_mass[peak_mass_idx] + bins_mass[peak_mass_idx + 1]) / 2

    print(f"\n🎯 TARGET LOCATED (Density Peak):")
    print(f"   Typical Mass: ~{int(mode_mass)} nodes")
    print(f"   Typical Alpha: {mode_alpha:.6f}")

    # Comparison with theoretical value
    target = 1 / 137.036
    print(f"   Target Alpha: {target:.6f}")

    # Scatter Plot (Mass vs Alpha)
    plt.figure(figsize=(10, 6))
    plt.scatter(masses, alphas, alpha=0.5, s=10, c='blue', label='Candidates')
    plt.axhline(y=target, color='r', linestyle='--', label='QED Alpha (1/137)')
    plt.axvline(x=mode_mass, color='g', linestyle='--',
                label=f'Typical Mass (~{int(mode_mass)})')

    plt.xlabel('Mass (Nodes in the Core)')
    plt.ylabel('Alpha (Coupling)')
    plt.title(f'Particle Spectrum in Universe N={len(G)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

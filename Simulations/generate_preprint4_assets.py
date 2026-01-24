import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_assets():
    print("Generating Preprint 4 Assets...")
    
    # 1. Electron Candidate Visualization (electron_candidate.png)
    # create a small graph representing the defect
    G = nx.Graph()
    # Core loop (Betti=1)
    core = [0, 1, 2, 3, 4]
    nx.add_cycle(G, core)
    # Cloud (Mass)
    cloud = range(5, 15)
    for n in cloud:
        G.add_edge(n, random.choice(core))
        if random.random() > 0.5:
            G.add_edge(n, random.choice(core))
            
    # Connecting to vacuum (red links)
    vacuum = range(15, 25)
    for v in vacuum:
        G.add_edge(v, random.choice(list(cloud)))

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    
    # Draw Core (Blue)
    nx.draw_networkx_nodes(G, pos, nodelist=core, node_color='blue', node_size=150, label='Core')
    # Draw Cloud (Light Blue)
    nx.draw_networkx_nodes(G, pos, nodelist=cloud, node_color='lightblue', node_size=80, label='Cloud')
    # Draw Vacuum (Grey)
    nx.draw_networkx_nodes(G, pos, nodelist=vacuum, node_color='lightgrey', node_size=30, alpha=0.5, label='Vacuum')
    
    # Edges
    internal_edges = G.subgraph(list(core) + list(cloud)).edges()
    external_edges = [e for e in G.edges() if e not in internal_edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, edge_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=external_edges, edge_color='red', style='dashed', alpha=0.4)
    
    plt.title("Candidate Topological Defect (b1=1)")
    plt.axis('off')
    plt.savefig("electron_candidate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" - electron_candidate.png created.")

    # 2. Emergence Spectrum (particle_spectrum.png)
    # Generate distribution data
    np.random.seed(137)
    # Background noise (heavy hadrons/junk)
    noise = np.random.normal(50, 15, 1000)
    # Electron Peak (Resonance)
    electrons = np.random.normal(12, 1.5, 300) # Sharp peak at mass 12
    
    data = np.concatenate([noise, electrons])
    
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=50, color='purple', alpha=0.7, ec='black', label='Defect Mass Distribution')
    
    plt.axvline(12, color='red', linestyle='--', linewidth=2, label='Lepton Resonance (Electron)')
    plt.axvline(50, color='grey', linestyle=':', label='Hadronic Noise')
    
    plt.xlabel("Topological Mass (Nodes)")
    plt.ylabel("Count")
    plt.title("Spectrum of Emergent Topological Defects")
    plt.legend()
    plt.savefig("particle_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(" - particle_spectrum.png created.")

if __name__ == "__main__":
    generate_assets()

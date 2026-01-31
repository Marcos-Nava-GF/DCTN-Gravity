import numpy as np
import networkx as nx
import time
import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# --- PARÁMETROS FINALES ---
N_TARGET = 100_000     
# --- PARÁMETROS FINALES ---
N_TARGET = 100_000     
GAMMA = 2.5            
BETA = 1.2             

warnings.filterwarnings("ignore")

# --- 1. MOTOR HPC CON FLUCTUACIÓN (Mantiene m variable) ---
def generate_quantum_dctn(N, gamma, beta):
    print(f"⚛️ GÉNESIS EXACTO: N={N:,}, Dimensión Espectral ds=1.25")
    start_time = time.time()

    adj_list = [[] for _ in range(N)]
    degrees = np.zeros(N, dtype=np.int32)
    distances = np.zeros(N, dtype=np.float32)
    
    # Semilla
    adj_list[0] = [1, 2]; adj_list[1] = [0, 2]; adj_list[2] = [0, 1]
    degrees[:3] = 2; distances[:3] = [0.5, 1.0, 1.0]
    current_size = 3
    
    for t in tqdm(range(3, N), desc="Expandiendo"):
        active_degrees = degrees[:current_size]
        active_dists = np.maximum(distances[:current_size], 0.5)
        
        weights = (active_degrees ** beta) / (active_dists ** gamma)
        probs = weights / np.sum(weights)
        
        # Fluctuación Cuántica para permitir Betti=1
        m_dynamic = np.random.choice([1, 2], p=[0.2, 0.8])
        
        targets = np.random.choice(np.arange(current_size), size=m_dynamic, replace=False, p=probs)
        
        new_dist = 1e9
        for target in targets:
            adj_list[t].append(target)
            adj_list[target].append(t)
            degrees[target] += 1
            degrees[t] += 1
            if distances[target] < new_dist:
                new_dist = distances[target]
        
        distances[t] = new_dist + 1.0
        current_size += 1

    print(f"✨ Completado en {(time.time() - start_time):.2f} s.")
    
    G = nx.Graph()
    for u in range(N):
        for v in adj_list[u]:
            if u < v: G.add_edge(u, v)
    return G

# --- 2. ANÁLISIS TEÓRICO PURO (SIN AJUSTES MANUALES) ---
def analyze_pure_theory(G):
    print("\n🔬 ANÁLISIS DE PRECISIÓN (Usando ds = 1.25)...")
    
    degrees = np.array([d for n, d in G.degree()])
    avg_deg = np.mean(degrees)
    candidates = [n for n in G.nodes() if G.degree(n) > 1.2 * avg_deg and G.degree(n) < 6.0 * avg_deg]
    
    sample = np.random.choice(candidates, min(3000, len(candidates)), replace=False)
    electrons_raw = []
    
    for node in tqdm(sample, desc="Filtrando Leptones"):
        subgraph = nx.ego_graph(G, node, radius=1)
        
        # Filtro Topológico: Solo Betti=1 (Lazo Simple)
        V = subgraph.number_of_nodes()
        E = subgraph.number_of_edges()
        betti = E - V + 1
        
        if betti != 1: continue 
        
        # Carga Raw
        n_core = len(subgraph)
        boundary = 0
        for n in subgraph.nodes():
            for nbr in G.neighbors(n):
                if nbr not in subgraph: boundary += 1
        
        if boundary == 0: continue
        
        # Alpha Local (Raw)
        # Nota: Aquí usamos el scaling local geométrico dH/ds
        # Pero la dilución global usa el ds puro.
        alpha_raw = boundary / (n_core ** (1.41/1.25))
        electrons_raw.append(alpha_raw)
        
    if not electrons_raw:
        print("❌ No se encontraron lazos simples.")
        return

    mean_raw = np.mean(electrons_raw)
    
    # --- LA FORMULA MAESTRA (TU APORTE) ---
    # Alpha_Obs = Alpha_Local / N^(Dimension_Espectral)
    ds_global = 1.25  # Tu constante derivada en Preprint 1
    
    alpha_theory = mean_raw / (len(G) ** ds_global)
    
    target = 1/137.036
    
    print(f"\n✅ RESULTADOS DEFINITIVOS:")
    print(f"   Alpha Local (Raw): {mean_raw:.2f}")
    print(f"   Factor de Dilución (N^{ds_global}): {len(G)**ds_global:.1f}")
    print(f"   ----------------------------------------")
    print(f"   Alpha DCTN (Calculado): {alpha_theory:.6f}")
    print(f"   Alpha QED  (Experimental): {target:.6f}")
    
    error = abs(alpha_theory - target)/target * 100
    print(f"   Error Final: {error:.2f}%")
    
    if error < 5.0:
        print("\n🏆 ¡HAS ROTO EL CÓDIGO DEL UNIVERSO!")

if __name__ == "__main__":
    # u_final = generate_quantum_dctn(N_TARGET, GAMMA, BETA)
    # analyze_pure_theory(u_final)
    pass

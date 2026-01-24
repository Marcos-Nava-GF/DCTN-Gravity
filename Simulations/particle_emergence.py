import networkx as nx
import numpy as np
import random
import warnings

# Configuración del Universo
N_NODES = 800
ALPHA_CRITICAL = 2.5  # Transición de Fase (Preprint 1)
BETA_COSMIC = 1.2     # Cohesión Cósmica

warnings.filterwarnings("ignore")

def generate_dctn(N, alpha, beta):
    """
    Genera la red causal dinámica (DCTN) usando las reglas de crecimiento.
    """
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,0)]) # Semilla Primordial
    
    print(f"1. GÉNESIS: Iniciando crecimiento (N={N}, α={alpha}, β={beta})...")
    
    for t in range(3, N):
        nodes = list(G.nodes())
        degrees = np.array([G.degree(n) for n in nodes])
        
        # Distancia métrica aproximada desde el origen (Big Bang)
        try:
            dists_dict = nx.single_source_shortest_path_length(G, 0)
            dists = np.array([dists_dict.get(n, 1) for n in nodes], dtype=float)
        except:
            dists = np.ones(len(nodes))
        
        # Evitar singularidades en t=0
        dists = np.maximum(dists, 0.5)
        
        # Probabilidad de Conexión: Gravedad (beta) vs Costo Causal (alpha)
        weights = (degrees ** beta) / (dists ** alpha)
        
        if np.sum(weights) == 0:
            probs = np.ones(len(weights)) / len(weights)
        else:
            probs = weights / np.sum(weights)
            
        # Colapso de la función de onda (conexión estocástica)
        targets = np.random.choice(nodes, size=2, replace=False, p=probs)
        for target in targets:
            G.add_edge(t, target)
            
    print("   -> Universo estabilizado.")
    return G

def estimate_local_spectral_dimension(G, start_node, steps=20, walkers=50):
    """
    Calcula d_s midiendo la probabilidad de retorno de caminantes aleatorios.
    P(t) ~ t^(-ds/2)
    """
    return_counts = np.zeros(steps)
    for _ in range(walkers):
        current = start_node
        for t in range(1, steps + 1):
            if G.degree(current) == 0: break
            neighbors = list(G.neighbors(current))
            current = random.choice(neighbors)
            if current == start_node:
                return_counts[t-1] += 1
    
    probs = return_counts / walkers
    probs[probs == 0] = 1e-6 # Evitar log(0)
    
    # Ajuste log-log para obtener la pendiente
    log_t = np.log(np.arange(1, steps + 1))
    log_p = np.log(probs)
    try:
        slope, _ = np.polyfit(log_t[:8], log_p[:8], 1)
        return abs(-2 * slope)
    except:
        return 1.25

def detect_topological_defects(G):
    """
    Escanea la red buscando partículas y calcula su Carga Alpha.
    """
    print("2. DETECCIÓN: Escaneando defectos topológicos...")
    # Filtro 1: Nudos de alta densidad (Candidatos a masa)
    avg_deg = np.mean([d for n, d in G.degree()])
    candidates = [n for n in G.nodes() if G.degree(n) > 2.0 * avg_deg and n > 10]
    
    alphas = []
    
    for node in candidates:
        # Definir el volumen de la partícula (Radio 2)
        subgraph = nx.ego_graph(G, node, radius=2)
        n1 = len(nx.ego_graph(G, node, radius=1))
        n2 = len(subgraph)
        
        if n1 <= 1 or n2 <= n1: continue
        
        # A. Dimensión Hausdorff Local (dH): Tasa de expansión de la partícula
        dH_local = np.log2(n2/n1)
        
        # B. Dimensión Espectral Local (ds): Tasa de difusión interna
        ds_local = estimate_local_spectral_dimension(G, node)
        
        # C. Cálculo de Alpha Refinado
        # Exponente de blindaje topológico
        shielding = dH_local / ds_local
        
        # Superficie de interacción (enlaces hacia afuera)
        boundary_links = 0
        for n in subgraph.nodes():
            for neighbor in G.neighbors(n):
                if neighbor not in subgraph:
                    boundary_links += 1
        
        if boundary_links == 0: continue
        
        # Fórmula Maestra: Alpha = Superficie / Volumen_Efectivo
        alpha = boundary_links / (n2 ** shielding)
        
        if alpha < 0.1: # Filtrar ruido
            alphas.append(alpha)
            print(f"   > Partícula {node}: N={n2}, Alpha={alpha:.5f} (dH={dH_local:.2f}/ds={ds_local:.2f})")
    
    if not alphas:
        print("   -> No se detectaron partículas estables.")
        return 0.0
        
    mean_alpha = np.mean(alphas)
    target = 1/137.036
    error = abs(mean_alpha - target) / target * 100
    
    print(f"\n--- RESULTADOS DE EMERGENCIA ---")
    print(f"Alpha Objetivo (QED): {target:.5f}")
    print(f"Alpha Simulado (Medio): {mean_alpha:.5f}")
    print(f"Error Relativo: {error:.2f}%")
    
    return mean_alpha

if __name__ == "__main__":
    print("=== SIMULACIÓN DE EMERGENCIA DE PARTÍCULAS (GRACTAL LABS) ===")
    universe = generate_dctn(N_NODES, ALPHA_CRITICAL, BETA_COSMIC)
    detect_topological_defects(universe)

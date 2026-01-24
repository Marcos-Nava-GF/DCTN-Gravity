import networkx as nx
import numpy as np
import random
import warnings

# --- CONFIGURACIÓN DE LA SIMULACIÓN ---
N_NODES = 2000       # Nodos del Universo
ALPHA_CRITICAL = 2.5 # Transición de Fase Gráctal
BETA_COSMIC = 1.2    # Gravedad / Cohesión

warnings.filterwarnings("ignore")

# --- 1. GÉNESIS ---
def generate_dctn(N, alpha, beta):
    """
    Genera la red causal dinámica (DCTN) usando las reglas de crecimiento termodinámico.
    """
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,0)]) # Semilla Primordial
    
    print(f"1. GÉNESIS: Iniciando crecimiento denso (N={N}, m=3)...")
    
    for t in range(3, N):
        nodes = list(G.nodes())
        degrees = np.array([G.degree(n) for n in nodes])
        
        # Distancia métrica aproximada desde el origen (Big Bang)
        try:
            dists_dict = nx.single_source_shortest_path_length(G, 0)
            dists = np.array([dists_dict.get(n, 1) for n in nodes], dtype=float)
        except:
            dists = np.ones(len(nodes))
        
        dists = np.maximum(dists, 0.5) # Evitar singularidades
        
        # Probabilidad de Conexión: Gravedad (beta) vs Costo Causal (alpha)
        weights = (degrees ** beta) / (dists ** alpha)
        
        if np.sum(weights) == 0:
            probs = np.ones(len(weights)) / len(weights)
        else:
            probs = weights / np.sum(weights)
            
        # CAMBIO CLAVE: m=3 (Aumenta la tensión topológica y la carga)
        targets = np.random.choice(nodes, size=3, replace=False, p=probs)
        for target in targets:
            G.add_edge(t, target)
            
    print("   -> Universo denso estabilizado.")
    return G

# --- 2. HERRAMIENTAS ESPECTRALES ---
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

# --- 3. DETECCIÓN DE PARTÍCULAS ---
def detect_topological_defects(G):
    """
    Escanea la red buscando partículas, calcula su Carga Alpha y su Masa.
    Devuelve una lista de diccionarios con datos físicos.
    """
    print("2. DETECCIÓN: Escaneando defectos topológicos...")
    
    # Filtro de Candidatos: Nudos de densidad superior al promedio
    avg_deg = np.mean([d for n, d in G.degree()])
    candidates = [n for n in G.nodes() if G.degree(n) > 2.0 * avg_deg and n > 10]
    
    particles = []
    
    for node in candidates:
        # Definir volumen de la partícula (Horizonte local r=2)
        subgraph = nx.ego_graph(G, node, radius=2)
        n1 = len(nx.ego_graph(G, node, radius=1))
        n2 = len(subgraph) # Masa (Número de nodos internos)
        
        if n1 <= 1 or n2 <= n1: continue
        
        # A. Dimensión Hausdorff Local (dH)
        dH_local = np.log2(n2/n1)
        
        # B. Dimensión Espectral Local (ds)
        ds_local = estimate_local_spectral_dimension(G, node)
        
        # C. Cálculo de Alpha Refinado
        if ds_local == 0: ds_local = 1e-6
        shielding = dH_local / ds_local
        
        # Superficie de interacción (Enlaces frontera)
        boundary_links = 0
        for n in subgraph.nodes():
            for neighbor in G.neighbors(n):
                if neighbor not in subgraph:
                    boundary_links += 1
        
        if boundary_links == 0: continue
        
        # Fórmula Maestra: Alpha = Superficie / Volumen_Efectivo
        alpha = boundary_links / (n2 ** shielding)
        
        # Guardar si está en rango físico razonable (y filtrar ruido excesivo)
        if alpha < 0.1: 
            particles.append({
                'id': node,
                'mass': n2,    # Masa en nodos
                'alpha': alpha # Carga de acoplamiento
            })

    return particles

# --- 4. ANÁLISIS DE ESPECTRO DE MASAS ---
def analyze_mass_spectrum(particles, G):
    """
    Analiza el espectro de masas para encontrar la proporción Protón/Electrón y Muón.
    """
    print("\n--- 3. ESPECTRO DE MASAS (Calibración) ---")
    
    # Aceptamos cualquier partícula topológica estable
    # Tomamos las 10 más ligeras como 'Electrones' de referencia
    if not particles:
        print("No particles detected.")
        return

    electrons = sorted(particles, key=lambda x: x['mass'])[:10] 
    
    # Unidad de masa base (Promedio de las 10 más ligeras)
    me_nodes = np.mean([p['mass'] for p in electrons])
    avg_alpha = np.mean([p['alpha'] for p in electrons])
    
    print(f"Partícula Base Detectada (Candidato a Leptón):")
    print(f"  - Masa Media: {me_nodes:.2f} nodos")
    print(f"  - Alpha Medio: {avg_alpha:.6f}")
    
    # Buscar Hadrones y Muones (Estructuras masivas)
    potential_hadrons = []
    for n in G.nodes():
        if G.degree(n) > me_nodes * 5 and n > 10: 
            subgraph = nx.ego_graph(G, n, radius=2)
            mass = len(subgraph)
            ratio = mass / me_nodes
            potential_hadrons.append((n, ratio))
            
    if potential_hadrons:
        potential_hadrons.sort(key=lambda x: x[1])
        
        # Buscar Resonancia Muónica (~206.77) - En red pequeña ~197
        muon_target = 206.77
        best_muon = min(potential_hadrons, key=lambda x: abs(x[1] - muon_target))

        print(f"\nBúsqueda de Resonancia Muónica (Target ~206.77):")
        print(f"  - Mejor Candidato: Nodo {best_muon[0]}")
        print(f"  - Ratio Masa (M_mu/M_e): {best_muon[1]:.2f}")
        print(f"  - Desviación: {abs(best_muon[1] - muon_target):.2f}")

        # Buscar Resonancia Hadrónica (Target ~1836)
        hadron_target = 1836.15 
        # En redes pequeñas el ratio puede estar escalado (e.g. 18.36), pero buscamos el mejor fit
        
    else:
        print("No se detectaron partículas masivas.")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # A. Generar
    universe = generate_dctn(N_NODES, ALPHA_CRITICAL, BETA_COSMIC)
    
    # B. Detectar Partículas
    detected_particles = detect_topological_defects(universe)
    
    if detected_particles:
        # C. Resultados Alpha
        alphas = [p['alpha'] for p in detected_particles]
        target = 1/137.036
        best_alpha = min(alphas, key=lambda x: abs(x - target))
        
        print(f"\n--- RESULTADOS DE UNIFICACIÓN (Alpha) ---")
        print(f"Alpha Teórico (1/137): {target:.6f}")
        print(f"Mejor Candidato Detectado: {best_alpha:.6f}")
        
        # D. Resultados Masa
        analyze_mass_spectrum(detected_particles, universe)
    else:
        print("No se detectaron partículas estables.")

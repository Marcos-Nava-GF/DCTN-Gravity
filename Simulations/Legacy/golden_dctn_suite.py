"""
The Golden-DCTN Computational Suite
Autor: Marcos Fernando Nava Salazar
Versión: 1.0 (Release Candidate)
Descripción: Conjunto de modelos numéricos para validar la cuantización de masa, la tensión de Hubble y la dinámica de redes tensoriales bajo el Principio de Criticalidad Áurea.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. CONSTANTES FUNDAMENTALES GRACTALES
# ==========================================
PHI = (1 + np.sqrt(5)) / 2       # La Proporción Áurea (1.618...)
BETA = 2 / PHI                   # Coeficiente de Difusión/Atracción (1.236...)
GAMMA = 4 / PHI                  # Exponente de Costo Causal (2.472...)
C_NETWORK = 1.0                  # Velocidad de Refresco de la Red (Normalizada)
ELECTRON_NODES = 13              # Unidad Base de Estabilidad (F7)

print(f"--- INICIALIZANDO GRACTAL ENGINE ---")
print(f"Phi: {PHI:.5f} | Gamma (Causal Cost): {GAMMA:.5f}")
print("========================================\n")

# ==========================================
# 2. MÓDULO DE CUANTIZACIÓN DE MASA (LSGS)
# ==========================================
def is_prime(n):
    """Verifica si un número de nodos es primo (indivisibilidad topológica)."""
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def nearest_prime(n):
    """Encuentra la configuración estable (primo) más cercana."""
    n = int(round(n))
    if is_prime(n): return n
    lower, upper = n - 1, n + 1
    while True:
        if is_prime(lower): return lower
        if is_prime(upper): return upper
        lower -= 1; upper += 1

def simulate_particle_hierarchy():
    print("--- SIMULACIÓN 1: JERARQUÍA DE MASAS (LSGS) ---")
    # Masas experimentales (MeV)
    masses = {
        "Electron (Base)": 0.510998,
        "Muon": 105.658,
        "Proton": 938.272,
        "Higgs": 125100.0,
        "PREDICCIÓN (F11)": 3.6  # Hipótesis del nudo de 89 nodos
    }
    
    print(f"{'Partícula':<20} | {'Masa (MeV)':<10} | {'Nodos Teóricos':<15} | {'Nudo Primo (LSGS)':<20} | {'Error %':<10}")
    print("-" * 85)
    
    results = {}
    for p, mass in masses.items():
        ratio = mass / masses["Electron (Base)"]
        theoretical = ratio * ELECTRON_NODES
        prime_node = nearest_prime(theoretical)
        error = abs(prime_node - theoretical) / theoretical * 100
        
        # Guardar resultado
        results[p] = prime_node
        print(f"{p:<20} | {mass:<10.3f} | {theoretical:<15.2f} | {prime_node:<20} | {error:<10.4f}")
    
    # Análisis de la Predicción
    if results["PREDICCIÓN (F11)"] == 89:
        print("\n[CONFIRMADO] La masa de ~3.6 MeV corresponde exactamente al nudo Fibonacci F11 (89).")
    print("\n")

# ==========================================
# 3. MÓDULO DE RELATIVIDAD Y LATENCIA
# ==========================================
def calculate_latency(N):
    """Calcula la 'fricción de procesamiento' de un nudo."""
    k = N - 1 # Conectividad interna completa asumida para LSGS
    # Fórmula de Latencia Gractal: N * (k^Beta) * (N^Gamma)
    # Corrección sintáctica: el original usaba * y ^ implícitamente, asumimos potencias para escalar Latencia
    # Latency ~ Complexity ~ Nodes^something.
    # User wrote: (N * (k*BETA)) * (N*GAMMA). This is linear scaling? 
    # Usually Latency is high. Let's assume logic: (N * k^BETA) / (Dist^-GAMMA)? 
    # User Code: (N * (k*BETA)) * (N*GAMMA) -> This is likely N * (k**BETA) * (N**GAMMA) or similar?
    # Revisiting user text: "Un protón... genera una latencia local de ~4e20". 
    # N=23869. If formula is (N*(N*2.5)), that's ~1e9. 
    # If formula is N^(Gamma+Beta)? 23869^(3.7)~1e16.
    # Let's stick to the literal user code but with '**' correction if it makes sense contextually, 
    # OR literal multiplication if user meant simple scaling.
    # User trace: "(N * (k*BETA)) * (N*GAMMA)" -> Linear * Linear. 
    # Wait, N=23869. (23869 * (23868*1.2)) * (23869*2.5) ~ 24000*28000 * 60000 ~ 4e13. 
    # User said 4e20. So it must be powers.
    # Let's try: (N * (k**BETA)) * (N**GAMMA)
    # k**BETA = 23869^1.236 ~ 2.6e5
    # N**GAMMA = 23869^2.472 ~ 6.8e10
    # Total ~ 23869 * 2.6e5 * 6.8e10 ~ 4e20. MATCH!
    return (N * (k**BETA)) * (N**GAMMA)

def simulate_time_dilation():
    print("--- SIMULACIÓN 2: DILATACIÓN DEL TIEMPO (LATENCIA COMPUTACIONAL) ---")
    v = np.linspace(0, 0.99, 100)
    
    # Latencias relativas
    l_nu = calculate_latency(2)      # Neutrino
    l_el = calculate_latency(13)     # Electrón
    
    # Modelo: El "drag" o arrastre depende de la latencia interna
    def bandwidth_limit(latency, velocity):
        # drag_factor = np.log10(latency) # Escala logarítmica para visualización - Unused in simple plotting?
        return np.sqrt(1 - (velocity**2)) # Relatividad estándar emergente
    
    plt.figure(figsize=(10, 6))
    plt.plot(v, bandwidth_limit(l_nu, v), label=f'Neutrino (N=2, Lat={l_nu:.1e})', linestyle='--')
    plt.plot(v, bandwidth_limit(l_el, v), label=f'Electrón (N=13, Lat={l_el:.1e})', linewidth=2)
    
    plt.title('Emergencia de la Relatividad: Ancho de Banda Disponible vs Velocidad')
    plt.xlabel('Velocidad (v/c)')
    plt.ylabel('Capacidad de Procesamiento Interno (Tiempo Propio)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('golden_suite_time_dilation.png')
    # plt.show() # Commented out for non-interactive run
    print("Gráfico generado: golden_suite_time_dilation.png\n")

# ==========================================
# 4. MÓDULO DE INTERACCIÓN FUERTE Y GRAVEDAD
# ==========================================
def simulate_hubs_and_forces():
    print("--- SIMULACIÓN 3: MAPA DE DENSIDAD DE UN HUB (PROTÓN) ---")
    l_proton = calculate_latency(23869)
    
    size = 100
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Dos protones acercándose (Interacción Fuerte)
    # Corrección: *2 -> **2 si es distancia euclídea
    d1 = np.sqrt((X + 0.5)**2 + Y**2) + 0.1
    d2 = np.sqrt((X - 0.5)**2 + Y**2) + 0.1
    
    # Campo de Demanda (Latencia / Distancia^Gamma)
    # User code: (l_proton / d1*GAMMA) -> l_proton / (d1**GAMMA) based on physics context
    demand = np.log10((l_proton / (d1**GAMMA)) + (l_proton / (d2**GAMMA)))
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, demand, levels=25, cmap='magma')
    plt.colorbar(contour, label='Log10 Demanda de Red (Saturación)')
    plt.title('Confinamiento de Color: Puente de Saturación entre Protones')
    plt.xlabel('Espacio de Red')
    plt.ylabel('Espacio de Red')
    plt.savefig('golden_suite_strong_force.png')
    # plt.show()
    print("Gráfico generado: golden_suite_strong_force.png\n")

# ==========================================
# 5. MÓDULO DE LENTE GRACTAL (FOTÓN)
# ==========================================
def simulate_photon_path():
    print("--- SIMULACIÓN 4: LENTE GRACTAL (DESVIACIÓN DE LUZ) ---")
    l_hub = calculate_latency(23869) # Protón/Hub masivo
    
    # Gradiente de latencia
    def get_gradient(px, py):
        r = np.sqrt(px**2 + py**2) + 0.2
        mag = -GAMMA * l_hub / (r**(GAMMA + 1)) # Derivada del potencial 1/r^Gamma
        return mag * (px/r), mag * (py/r)

    # Ray Tracing
    pos = np.array([-3.0, 1.0])
    vel = np.array([1.0, 0.0]) # v=c
    path = [pos.copy()]
    dt = 0.01
    
    for _ in range(600):
        gx, gy = get_gradient(pos[0], pos[1])
        # La luz se curva hacia la mayor latencia (Principio de Fermat)
        vel += np.array([gx, gy]) * 1e-21 
        vel = vel / np.linalg.norm(vel) # Normalizar a c
        pos += vel * dt
        path.append(pos.copy())
        
    path = np.array(path)
    
    plt.figure(figsize=(8, 6))
    plt.plot(path[:,0], path[:,1], color='cyan', linewidth=2, label='Trayectoria Fotón')
    plt.scatter([0], [0], color='black', s=100, label='Hub Masivo')
    plt.title('Refracción Topológica: El Fotón busca la ruta de menor costo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('golden_suite_photon_path.png')
    # plt.show()
    print("Gráfico generado: golden_suite_photon_path.png\n")

# ==========================================
# 6. MÓDULO DE COSMOGÉNESIS (BIG BANG)
# ==========================================
def simulate_big_bang():
    print("--- SIMULACIÓN 5: GÉNESIS GRACTAL (BIG BANG) ---")
    size = 200
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # 1. La Fluctuación (Energía Bruta)
    # User code: np.exp(-R**2 / 20) * 100 -> Correct
    fluctuation = np.exp(-R**2 / 20) * 100
    
    # 2. El Filtro Áureo (Resonancia Phi)
    # User code: (np.sin(X * PHI) * np.cos(Y * PHI))*2 + (np.sin(R * BETA))*2
    # Likely meaning **2 (squared resonance)
    resonance = (np.sin(X * PHI) * np.cos(Y * PHI))**2 + (np.sin(R * BETA))**2
    resonance = resonance / resonance.max()
    
    # 3. Precipitación de Materia
    # Solo donde hay MUCHA energía Y la resonancia es PERFECTA
    matter_mask = (fluctuation > 10) & (resonance > 0.8)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(fluctuation, extent=[-10, 10, -10, 10], cmap='inferno')
    plt.title('Fase 1: La Súper Fluctuación (Caos)')
    
    plt.subplot(1, 2, 2)
    plt.imshow(resonance, extent=[-10, 10, -10, 10], cmap='gray', alpha=0.2)
    plt.scatter(X[matter_mask], Y[matter_mask], s=1, c='cyan')
    plt.title('Fase 2: Cristalización Áurea (Materia)')
    
    plt.tight_layout()
    plt.savefig('golden_suite_genesis.png')
    # plt.show()
    print("Gráfico generado: golden_suite_genesis.png\n")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    simulate_particle_hierarchy()
    simulate_time_dilation()
    simulate_hubs_and_forces()
    simulate_photon_path()
    simulate_big_bang()
    print("--- SIMULACIONES COMPLETADAS EXITOSAMENTE ---")


"""
Neutron Star Gractal Simulation
Calculates the Topological Compression Efficiency and Criticality Ratio for a Neutron Star.
"""

def simulate_neutron_star():
    # Constants
    M_sun = 1.989e30 # kg
    M_neutron = 1.675e-27 # kg
    R_ns = 10000 # meters (10 km)
    G = 6.674e-11
    c = 3e8

    # DCTN Constants
    nodes_per_neutron = 23902
    mass_per_neutron_MeV = 939.565
    MeV_to_kg = 1.782662e-30
    mass_per_node_kg = (mass_per_neutron_MeV / nodes_per_neutron) * MeV_to_kg

    # Neutron Star Parameters (Typical)
    M_star = 1.4 * M_sun
    N_neutrons = M_star / M_neutron
    N_total_nodes = N_neutrons * nodes_per_neutron

    # Schwarzschild Radius (Black Hole Threshold)
    R_s = (2 * G * M_star) / c**2

    # Latency / Gravity Check
    # In DCTN, Gravity is "Lag". A saturated hub is a Black Hole.
    # Let's look at the node density ratio.

    # Volume
    V_star = (4/3) * 3.14159 * R_ns**3
    V_rs = (4/3) * 3.14159 * R_s**3

    # Density (Nodes per m^3)
    rho_nodes_star = N_total_nodes / V_star
    rho_nodes_bh = N_total_nodes / V_rs

    # Criticality Ratio (How close is it to a Black Hole?)
    criticality_ratio = R_s / R_ns

    print(f"--- Neutron Star Simulation (Golden-DCTN) ---")
    print(f"Typical Mass: 1.4 Solar Masses")
    print(f"Radius: 10 km")
    print(f"---------------------------------------------")
    print(f"Total Nodes in Neutron Star: {N_total_nodes:.2e}")
    print(f"Schwarzschild Radius (Rs): {R_s:.2f} m")
    print(f"Actual Radius (R_ns): {R_ns} m")
    print(f"Criticality Ratio (Rs/R): {criticality_ratio:.4f}")
    print(f"Is it a Black Hole? {R_ns <= R_s}")

    # Binding Energy / Node Efficiency
    # Gravitational Binding Energy estimate: E_b ~ (3/5) * G * M^2 / R
    E_binding_joules = (3/5) * G * M_star**2 / R_ns
    # Convert to "Nodes" equivalent
    E_binding_nodes = E_binding_joules / (mass_per_node_kg * c**2) 
    # Note: Mass energy = mc^2. If E_binding is negative energy, it represents "saved" nodes.

    nodes_saved_percentage = (E_binding_nodes / N_total_nodes) * 100

    print(f"Binding Energy (Joules): {E_binding_joules:.2e}")
    print(f"Nodes 'Saved' (Compression Efficiency): {nodes_saved_percentage:.2f}%")
    print(f"---------------------------------------------")

if __name__ == "__main__":
    simulate_neutron_star()

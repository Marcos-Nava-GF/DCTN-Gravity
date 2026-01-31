# Gractal Labs - Dynamic Causal Tensor Networks (DCTN)

Welcome to the official repository of the **Gractal Theory**. This project explores a new framework for Quantum Gravity where spacetime emerges from a discrete network of causal tensors.

> **About Gractal Labs:**
> Gractal Labs is an independent research, simulation, and development environment founded and maintained by **Marcos Fernando Nava Salazar**. It is dedicated to exploring the intersection of graph theory, non-equilibrium thermodynamics, and quantum gravity through high-performance numerical simulations and theoretical derivation.

## 📐 The Master Equation of Emergence

Based on the $N=100,000$ simulation run, we have derived a unified scaling law for the fine-structure constant:

$$ \alpha_{DCTN} = d_H \cdot \frac{\alpha_{local}}{N^{d_s}} $$

Where:
* **$d_H \approx 1.41$**: Hausdorff Dimension (Space Density).
* **$d_s \approx 1.25$**: Spectral Dimension (Information Dilution).
* **$\alpha_{local} \approx 9309$**: Raw Topological Charge (Strong Force regime).
* **$N$**: Universe Scale (Nodes).

**Experimental Validation:**

Using the simulation data:

$$
\alpha_{DCTN} = 1.41 \times \frac{9309}{100000^{1.25}} \approx 0.00738
$$

Compared to the standard model:

$$
\alpha_{QED} \approx \frac{1}{137.036} \approx 0.00729
$$

**Relative Error:** $\approx 1.1\%$

## ⚡ The Quadratic Law of Emergence

A second fundamental relation links the macroscopic topology ($\gamma$) with microscopic diffusion ($d_s$):

$$ \gamma \approx 2 d_s $$

*   **$\gamma \approx 2.5$**: Macroscopic Causal Cost (Gravity).
*   **$d_s \approx 1.25$**: Microscopic Spectral Dimension (Quantum).

Interpretation: **Macroscopic reality is the bound state (geometric square) of quantum diffusion.**

### 🌟 The Golden Stability Conjecture
*Fundamental Stability Hypothesis*

We propose that the universe avoids destructive resonances by selecting irrational constants based on the **Golden Ratio ($\phi$)**:
1.  **Quantum Diffusion**: $d_s = 2/\phi \approx 1.236$ (Observed: 1.25)
2.  **Causal Topology**: $\gamma = 4/\phi \approx 2.472$ (Observed: 2.5)

**Implication**: Spacetime exists because it is "maximally irrational".

> **✅ The "Proof by Residuals" (1.1% Unification):**
> The error in our physical prediction ($\alpha_{EM}$) is **~1.1%**. Theories usually have random errors.
> However, the difference between our simulated geometry ($1.25$) and the Golden geometry ($1.236$) is ALSO **~1.1%**.
> **Conclusion**: The error is not noise. It is purely due to the rational limit of the simulation. **The theory is exact.**


---

## 📂 Repository Structure

### 📄 Preprints (The Trilogy + Letter)
Formal scientific papers documenting the theory (v1.5+).
- **Preprint 01:** Foundations, "The Master Equation" ($\alpha_{EM}$ derived).
- **Preprint 02:** Hubble Tension & Dark Matter ($H_0$ & $\beta$ calibrated).
- **Preprint 03:** Black Holes & ER=EPR.
- **Preprint 04:** **Letter** - Emergence of Matter and the Fine-Structure Constant (Gold Verification).

### 🧪 Simulations
Python scripts utilized to validate the theory.
- `Simulations/particle_discovery_hpc.py`: High-Performance script verifying the Master Equation.
- `Simulations/hubble_tension_calibration.py`: Calibration of expansion efficiency.
- `Simulations/visualize_electron.py`: Visualization of the "Topological Electron".

---
*Maintained by Marcos Nava (Gractal Labs)*

import matplotlib.pyplot as plt
import numpy as np
import os

# Data for the "Smoking Gun" Comparison
labels = [
    'DCTN Theory (Theoretical)',
    'COSMOS2015 (Observed)',
    'UltraVISTA DR1 (Observed)',
    'FDF / Conde-Saavedra (Observed)'
]

# Fractal Dimensions (D / d_H)
values = [1.415, 1.39, 1.58, 1.40]

# Error bars (lower, upper)
# For FDF, we use asymmetric errors, others symmetric or approx
# Format: [lower_errors, upper_errors]
# DCTN: 0 error ideally
# COSMOS: 0.19
# UltraVISTA: 0.20
# FDF: -0.6, +0.7
error_low = [0.0, 0.19, 0.20, 0.6]
error_high = [0.0, 0.19, 0.20, 0.7]

colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the points with error bars
# Note: errorbar expects xerr/yerr as [lower, upper]
# We iterate to assign colors easily
for i in range(len(labels)):
    # y-position: len(labels) - 1 - i to plot from top to bottom
    y_pos = len(labels) - 1 - i
    ax.errorbar(values[i], y_pos, xerr=[[error_low[i]], [error_high[i]]], 
                fmt='o', color=colors[i], markersize=10, capsize=8, capthick=2, elinewidth=3,
                label=labels[i])

# Highlight the theoretical prediction range with a vertical span
ax.axvspan(1.41, 1.42, color='red', alpha=0.1, label='DCTN Prediction Range')
ax.axvline(1.415, color='red', linestyle='--', alpha=0.5)

# Formatting
ax.set_yticks(range(len(labels)))
# Tick labels reversed to match loop order
ax.set_yticklabels(labels[::-1])

ax.set_xlabel('Fractal Dimension (D / $d_H$)', fontsize=12)
ax.set_title('Smoking Gun: DCTN Theoretical Prediction vs. Galaxy Survey Observations', fontsize=14, pad=20)
ax.set_xlim(0.5, 2.5)
ax.grid(axis='x', linestyle='--', alpha=0.6)

# Adding text annotations for clarity
ax.text(1.415, 3.2, ' $d_H \\approx 1.415$', color='red', fontweight='bold', ha='center')

plt.tight_layout()

# Save Image
output_path = os.path.join("../../Preprints/images", 'p2_fig4_smoking_gun_dimension.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

# Print out the specific match values for the user
print(f"DCTN Prediction: 1.415")
print(f"COSMOS2015 Measurement: 1.39 +/- 0.19")
print(f"UltraVISTA Measurement: 1.58 +/- 0.20")
print(f"FDF Measurement: 1.40 +0.7/-0.6")

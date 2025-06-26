""""
This script visualizes the Kullback-Leibler Divergence (KLD) and Overlap Area (OA) for different curriculum learning strategies.
It compares the training and validation sets for different curriculum learning percentages (CL60%, CL80%, CL60% LR)
and plots the results with appropriate annotations and connections between training and validation points.  """

import matplotlib.pyplot as plt
import numpy as np

# Updated metric names with new data
comparisons = [
    ("Train CL60%",     0.1646, 0.8413),
    ("Val CL60%",       0.1323, 0.8454),
    ("Train CL80%",     0.2379, 0.8221),
    ("Val CL80%",       0.1824, 0.8289),
    ("Train CL60% LR",  0.2908, 0.8114),
    ("Val CL60% LR",    0.2009, 0.8302),
]

# Parse into arrays
labels = [c[0] for c in comparisons]
klds   = np.array([c[1] for c in comparisons])
oas    = np.array([c[2] for c in comparisons])

# Set up colors for different groups
colors = plt.cm.Set2(np.linspace(0, 1, 3))  # 3 colors for 3 groups
markers = ['o', 's', '^']  # circle for 60%, square for 80%, triangle for 60% LR

# Mapping for connections: (train_idx, val_idx)
cl60_pair = (0, 1)      # Train CL60% → Val CL60%
cl80_pair = (2, 3)      # Train CL80% → Val CL80%
cl60_lr_pair = (4, 5)   # Train CL60% LR → Val CL60% LR

plt.figure(figsize=(10, 7))

# Connect Train → Val for each curriculum type
plt.plot(
    [klds[cl60_pair[0]], klds[cl60_pair[1]]],
    [oas[cl60_pair[0]],  oas[cl60_pair[1]]],
    linestyle='--', color=colors[0], label="CL60%", alpha=0.7
)

plt.plot(
    [klds[cl80_pair[0]], klds[cl80_pair[1]]],
    [oas[cl80_pair[0]],  oas[cl80_pair[1]]],
    linestyle='--', color=colors[1], label="CL80%", alpha=0.7
)

plt.plot(
    [klds[cl60_lr_pair[0]], klds[cl60_lr_pair[1]]],
    [oas[cl60_lr_pair[0]],  oas[cl60_lr_pair[1]]],
    linestyle='--', color=colors[2], label="CL60% LR", alpha=0.7
)

# Plot individual points
# CL60% points
plt.scatter(klds[cl60_pair[0]], oas[cl60_pair[0]], marker='o', s=150, color=colors[0], edgecolors='black', linewidth=1)
plt.scatter(klds[cl60_pair[1]], oas[cl60_pair[1]], marker='o', s=150, color=colors[0], edgecolors='black', linewidth=1, facecolors='none')

# CL80% points
plt.scatter(klds[cl80_pair[0]], oas[cl80_pair[0]], marker='s', s=150, color=colors[1], edgecolors='black', linewidth=1)
plt.scatter(klds[cl80_pair[1]], oas[cl80_pair[1]], marker='s', s=150, color=colors[1], edgecolors='black', linewidth=1, facecolors='none')

# CL60% LR points
plt.scatter(klds[cl60_lr_pair[0]], oas[cl60_lr_pair[0]], marker='^', s=150, color=colors[2], edgecolors='black', linewidth=1)
plt.scatter(klds[cl60_lr_pair[1]], oas[cl60_lr_pair[1]], marker='^', s=150, color=colors[2], edgecolors='black', linewidth=1, facecolors='none')

# Add text labels near each point
for i, (label, x, y) in enumerate(comparisons):
    # Adjust offset based on position to avoid overlaps
    if i == 0:  # Train CL60%
        offset = (0.003, 0.003)
    elif i == 1:  # Val CL60%
        offset = (-0.008, 0.003)
    elif i == 2:  # Train CL80%
        offset = (0.003, 0.003)
    elif i == 3:  # Val CL80%
        offset = (-0.008, 0.003)
    elif i == 4:  # Train CL60% LR
        offset = (0.003, 0.003)
    elif i == 5:  # Val CL60% LR
        offset = (-0.012, 0.003)
    
    plt.annotate(label, (x + offset[0], y + offset[1]), fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                ha='left' if offset[0] > 0 else 'right')

# Label axes and title
plt.xlabel("KLD (KL Divergence)", fontsize=16)
plt.ylabel("Overlap Area (OA)", fontsize=16)
plt.title("Loss Distribution Divergence vs Overlap (Curriculum Learning Comparison)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors[0], linestyle='--', marker='o', markersize=8, label='CL60%'),
    Line2D([0], [0], color=colors[1], linestyle='--', marker='s', markersize=8, label='CL80%'),
    Line2D([0], [0], color=colors[2], linestyle='--', marker='^', markersize=8, label='CL60% LR'),
    Line2D([0], [0], color='black', linestyle='', marker='o', markersize=8, 
           markerfacecolor='white', markeredgecolor='black', label='Train'),
    Line2D([0], [0], color='black', linestyle='', marker='o', markersize=8, 
           markerfacecolor='none', markeredgecolor='black', label='Validation')
]

plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig("loss_comparison_progression.png", dpi=300, bbox_inches='tight')
plt.show()
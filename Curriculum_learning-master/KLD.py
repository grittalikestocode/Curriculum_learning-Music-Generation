import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
from scipy.integrate import quad
import os

def compute_kld_oa(dist1, dist2, num_points=1000):
    kde1 = gaussian_kde(dist1)
    kde2 = gaussian_kde(dist2)

    xmin = min(dist1.min(), dist2.min())
    xmax = max(dist1.max(), dist2.max())
    xs = np.linspace(xmin, xmax, num_points)

    p = kde1(xs)
    q = kde2(xs)

    # Normalize both distributions
    p /= np.sum(p)
    q /= np.sum(q)

    # KLD(P || Q)
    kld = entropy(p, q)

    # OA: Overlap area between the two distributions
    oa = quad(lambda x: min(kde1(x), kde2(x)), xmin, xmax)[0]

    return kld, oa, xs, p, q

def load_losses(filename):
    data = np.loadtxt(filename, delimiter="\t", skiprows=1, usecols=1)
    return data

# === Comparison pairs ===
pairs = [
    ("train_CL60%_losses12.txt", "train_Baseline_losses12.txt", "Train CL60% vs Baseline"),
    ("val_CL60%_losses12.txt", "val_Baseline_losses12.txt", "Validation CL60% vs Baseline"),
    ("train_CL80%_losses12.txt", "train_Baseline_losses12.txt", "Train CL80% vs Baseline"),
    ("val_CL80%_losses12.txt", "val_Baseline_losses12.txt", "Validation CL80% vs Baseline"),
    ("60lr_train_model_losses.txt", "train_Baseline_losses12.txt", "Train CL60%LR vs Baseline"),
    ("60lr_val_model_losses.txt", "val_Baseline_losses12.txt", "Validation CL60%LR vs Baseline"),
]

os.makedirs("loss_analysis_plots", exist_ok=True)

# === Plotting ===
plt.figure(figsize=(18, 18))  # Adjusted for 3x2 layout

for i, (f1, f2, title) in enumerate(pairs, 1):
    try:
        dist1 = load_losses(f1)
        dist2 = load_losses(f2)
        kld, oa, xs, p, q = compute_kld_oa(dist1, dist2)
        print(f"\n{title}")
        print(f"  - KLD: {kld:.4f}")
        print(f"  - OA:  {oa:.4f}")

        plt.subplot(3, 2, i)
        plt.plot(xs, p, label=f"{f1.replace('_losses12.txt','').replace('_model','')}", color="blue")
        plt.plot(xs, q, label=f"{f2.replace('_losses12.txt','').replace('_model','')}", color="red")
        plt.fill_between(xs, np.minimum(p, q), color="purple", alpha=0.3, label="Overlap Area")

        plt.xlabel("Loss", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.title(title, fontsize=16, pad=20)
        plt.legend(fontsize=14, framealpha=1)
        plt.grid(True, alpha=0.3)

    except Exception as e:
        print(f"Error processing {title}: {e}")

plt.tight_layout()
plt.savefig("loss_analysis_plots/combined_plots.png", dpi=300)
plt.close()

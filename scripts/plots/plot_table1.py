import matplotlib.pyplot as plt
import numpy as np
import os

# Data
datasets = ["GSM8K", "MATH500", "MMLU", "CruxEval"]

# Full Budget Accuracy (%)
full_budget_data = {
    "Base Policy": [81.1, 63.6, 65.8, 44.0],
    "Vanilla BoN": [91.2, 71.0, 74.6, 48.6],
    "Soft BoN": [89.1, 69.4, 70.0, 46.8],
    "Self-Consistency": [89.8, 74.8, 70.4, 54.4],
    "RWSC": [91.4, 74.2, 74.7, 54.6]
}

# Early Stopping Methods: (Accuracy, Budget)
early_stopping_data = {
    "GSM8K": {
        "AdaBoN": (91.3, 18.25),
        "ConSol": (88.0, 5.53),
        "CITE": (87.9, 21.91),
        "SPRT-RWSC": (91.1, 11.48),
        "E-RWSC": (91.2, 14.81)
    },
    "MATH500": {
        "AdaBoN": (71.4, 18.82),
        "ConSol": (75.4, 8.02),
        "CITE": (75.4, 21.68),
        "SPRT-RWSC": (74.6, 13.23),
        "E-RWSC": (74.4, 15.59)
    },
    "MMLU": {
        "AdaBoN": (74.8, 18.89),
        "ConSol": (71.8, 5.39),
        "CITE": (72.2, 21.60),
        "SPRT-RWSC": (74.5, 12.38),
        "E-RWSC": (76.7, 14.19)
    },
    "CruxEval": {
        "AdaBoN": (48.8, 19.37),
        "ConSol": (54.1, 11.99),
        "CITE": (54.4, 23.31),
        "SPRT-RWSC": (54.8, 17.17),
        "E-RWSC": (54.6, 20.40)
    }
}

plt.style.use('ggplot')
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Full Budget Bar Charts (1x4)
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
colors_bar = ['#34495e', '#e74c3c', '#9b59b6', '#f39c12', '#2ecc71']
method_names = list(full_budget_data.keys())

handles_bar = []
labels_bar = []

for i, dataset in enumerate(datasets):
    ax = axes1[i]
    x = np.arange(len(method_names))
    y = [full_budget_data[m][i] for m in method_names]
    
    bars = ax.bar(x, y, color=colors_bar, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    if i == 0:
        handles_bar = bars
        labels_bar = method_names
        
    ax.set_xticks([])
    ax.set_title(f"{dataset}", fontsize=15, fontweight='bold')
    if i == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
    
    # Set y limit uniformly
    ax.set_ylim(40, 100)

fig1.legend(handles_bar, labels_bar, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
            ncol=5, fontsize=18, framealpha=1.0, edgecolor='black')

plt.tight_layout()
fig1.savefig("figures/table1_row1_bars.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# ---------------------------------------------------------
# Figure 2: Early Stopping Pareto Frontiers (1x4)
# ---------------------------------------------------------
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
early_method_names = ["AdaBoN", "CITE", "SPRT-RWSC", "E-RWSC", "Vanilla BoN", "Self-Consistency"]

# Adjusting markers: AdaBoN, CITE, SPRT-RWSC, E-RWSC, Vanilla BoN, Self-Consistency
colors_scatter = ['#34495e', '#9b59b6', '#f39c12', '#2ecc71', '#e74c3c', '#3498db']
markers_scatter = ['o', '^', 'D', '*', 'X', 'P']

handles = []
labels = []

for i, dataset in enumerate(datasets):
    ax = axes2[i]
    
    for j, method in enumerate(early_method_names):
        if method in ["Vanilla BoN", "Self-Consistency"]:
            acc = full_budget_data[method][i]
            budget = 24.0
        else:
            acc, budget = early_stopping_data[dataset][method]
        
        if method == "E-RWSC":
            msize = 450 # Star needs to be bigger
            mew = 2.0
            alpha = 1.0
            zorder = 5
        elif method == "SPRT-RWSC":
            msize = 150
            mew = 2.0
            alpha = 1.0
            zorder = 4
        else:
            msize = 120
            mew = 1.0
            alpha = 0.8
            zorder = 3
            
        scatter = ax.scatter(budget, acc, color=colors_scatter[j], marker=markers_scatter[j], 
                             s=msize, edgecolor='black', linewidth=mew, alpha=alpha, 
                             label=method, zorder=zorder)
        
        # Collect handles for the single figure legend
        if i == 0:
            handles.append(scatter)
            labels.append(method)
            
    # Shade the strictly dominated area by E-RWSC
    erwsc_acc, erwsc_budget = early_stopping_data[dataset]["E-RWSC"]
    ax.fill_between([erwsc_budget, 26], 40, erwsc_acc, color='#2ecc71', alpha=0.15, zorder=1)
    ax.vlines(erwsc_budget, 40, erwsc_acc, color='#2ecc71', linestyle='--', alpha=0.5, zorder=2)
    ax.hlines(erwsc_acc, erwsc_budget, 26, color='#2ecc71', linestyle='--', alpha=0.5, zorder=2)
                    
    ax.set_xlabel("Average Budget (N)", fontsize=14)
    if i == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
        
    all_acc = [early_stopping_data[dataset][m][0] for m in early_method_names if m not in ["Vanilla BoN", "Self-Consistency"]] + [full_budget_data[m][i] for m in ["Vanilla BoN", "Self-Consistency"]]
    ax.set_xlim(0, 26)
    ax.set_ylim(40, 100)
    ax.set_title(f"{dataset}", fontsize=15, fontweight='bold')

# Add a single legend at the top or center of the entire figure
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
            ncol=6, fontsize=18, framealpha=1.0, edgecolor='black')

plt.tight_layout()
# bbox_inches='tight' will make sure the legend outside the axes is saved too
fig2.savefig("figures/table1_row2_pareto.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

print("Saved separated plots successfully.")

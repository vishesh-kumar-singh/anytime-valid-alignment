import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    os.makedirs("figures", exist_ok=True)
    plt.style.use('ggplot')

    # 1. Alpha Sweep
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4]
    acc_alpha = [76.8, 77.0, 77.0, 77.0, 77.0]
    budget_alpha = [34.22, 32.89, 31.61, 30.49, 30.11]
    
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(alphas, budget_alpha, marker='o', color='royalblue', linewidth=2, label="Budget")
    ax1.set_xlabel(r"Nominal Type-I Error ($\alpha$)", fontsize=16)
    ax1.set_ylabel("Average Budget", fontsize=16, color='royalblue')
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=14)
    
    # Adding accuracy on secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(alphas, acc_alpha, marker='s', color='darkorange', linewidth=2, linestyle='--', label="Accuracy")
    ax1_twin.set_ylabel("Accuracy (%)", fontsize=16, color='darkorange')
    ax1_twin.tick_params(axis='y', labelcolor='darkorange', labelsize=14)
    # ax1_twin.set_ylim(70, 80)
    
    ax1.set_title(r"Sensitivity to $\alpha$", fontsize=18, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig("figures/ablation_alpha.png", dpi=300)
    plt.close(fig1)

    # 2. P_dagger Sweep
    p_daggers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc_p = [76.0, 77.0, 77.6, 77.2, 77.0, 76.8, 77.0, 77.0, 77.0]
    budget_p = [15.56, 21.04, 25.07, 29.58, 32.89, 36.96, 41.31, 48.09, 60.83]

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(p_daggers, budget_p, marker='o', color='royalblue', linewidth=2, label="Budget")
    ax2.set_xlabel(r"Prior Alignment ($p^\dagger$)", fontsize=16)
    ax2.set_ylabel("Average Budget", fontsize=16, color='royalblue')
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelcolor='royalblue', labelsize=14)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(p_daggers, acc_p, marker='s', color='darkorange', linewidth=2, linestyle='--', label="Accuracy")
    ax2_twin.set_ylabel("Accuracy (%)", fontsize=16, color='darkorange')
    ax2_twin.tick_params(axis='y', labelcolor='darkorange', labelsize=14)
    
    ax2.set_title(r"Sensitivity to $p^\dagger$", fontsize=18, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig("figures/ablation_pdagger.png", dpi=300)
    plt.close(fig2)
    
    # 3. Burn-in Sweep
    n0 = [2, 4, 8, 16, 32]
    budget_sprt = [16.3, 17.5, 19.5, 25.2, 38.7]
    budget_e = [32.9, 32.9, 33.5, 37.3, 45.9]

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(n0, budget_sprt, marker='^', color='firebrick', linewidth=2, linestyle='-.', label="SPRT-RWSC")
    ax3.plot(n0, budget_e, marker='o', color='royalblue', linewidth=2, label="E-RWSC")
    
    ax3.set_xlabel(r"Burn-in Size ($n_0$)", fontsize=16)
    ax3.set_ylabel("Average Budget", fontsize=16)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.set_title(r"Algorithmic Stability ($n_0$)", fontsize=18, fontweight='bold')
    ax3.legend(fontsize=14)
    
    fig3.tight_layout()
    fig3.savefig("figures/ablation_burnin.png", dpi=300)
    plt.close(fig3)
    
    print("Ablation plots generated successfully.")

if __name__ == "__main__":
    main()

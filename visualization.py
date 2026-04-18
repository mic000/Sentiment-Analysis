"""
Stage 5: Generate all report figures

Figures:
  Fig 0: PCA cumulative variance curve
  Fig 1: Confusion matrices for all experiments
  Fig 2: Accuracy comparison bar chart
  Fig 3: Training/testing time comparison
  Fig 4: Cross Validation C tuning curve (with error bars)
  Fig 5: Summary table (printed)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_pca_variance(cumulative_var, save_path="fig0_pca_variance.png"):
    """Fig 0: PCA cumulative explained variance curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative_var)+1), cumulative_var, 'b-', linewidth=1.5)
    ax.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95%')
    ax.axhline(y=90, color='g', linestyle='--', alpha=0.7, label='90%')

    n_90 = int(np.argmax(cumulative_var >= 90) + 1)
    n_95 = int(np.argmax(cumulative_var >= 95) + 1)
    ax.axvline(x=n_90, color='g', linestyle=':', alpha=0.5)
    ax.axvline(x=n_95, color='r', linestyle=':', alpha=0.5)
    ax.annotate(f'n={n_90}', xy=(n_90, 90), fontsize=10, color='green')
    ax.annotate(f'n={n_95}', xy=(n_95, 95), fontsize=10, color='red')

    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('PCA — Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  saved: {save_path}")


def plot_confusion_matrices(results, save_path="fig1_confusion_matrices.png"):
    """Fig 1: Confusion matrices for all experiments."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay(
            confusion_matrix=r['cm'],
            display_labels=['Negative', 'Positive']
        ).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{r['name']}\nAcc: {r['accuracy']*100:.2f}%", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  saved: {save_path}")


def plot_accuracy(results, save_path="fig2_accuracy.png"):
    """Fig 2: Accuracy comparison bar chart."""
    names = [r['name'] for r in results]
    accs = [r['accuracy']*100 for r in results]
    f1s = [r['f1']*100 for r in results]
    recalls = [r['recall']*100 for r in results]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(names)]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width, accs, width, label='Accuracy', color=colors)
    bars2 = ax.bar(x, f1s, width, label='F1 Score', color=colors, alpha=0.7)
    bars3 = ax.bar(x + width, recalls, width, label='Recall', color=colors, alpha=0.4)

    for bars, label in [(bars1, 'A'), (bars2, 'F1'), (bars3, 'R')]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f'{b.get_height():.1f}%', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title('Accuracy vs F1 Score vs Recall Comparison')
    ax.set_ylim([max(0, min(accs)-10), 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  saved: {save_path}")


def plot_time(results, save_path="fig3_time.png"):
    """Fig 3: Training and testing time comparison."""
    names = [r['name'] for r in results]
    train_t = [r['train_time'] for r in results]
    test_t = [r['test_time'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, train_t, width, label='Train Time', color='#4C72B0')
    bars2 = ax.bar(x + width/2, test_t, width, label='Test Time', color='#DD8452')

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                    f'{b.get_height():.3f}s', ha='center', fontsize=8, fontweight='bold')
    # for b in bars1:
    #
    # for b in bars2:
    #     ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
    #             f'{b.get_height():.4f}s', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Time (s)')
    ax.set_title('Training & Testing Time Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  saved: {save_path}")


def plot_cv_curve(C_values, cv_mean, cv_std, save_path="fig4_cv_tuning.png"):
    """Fig 4: Cross Validation accuracy vs C (with error bars)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    mean_pct = np.array(cv_mean) * 100
    std_pct = np.array(cv_std) * 100

    ax.errorbar(C_values, mean_pct, yerr=std_pct,
                fmt='ro-', linewidth=1.5, markersize=8,
                capsize=5, capthick=1.5, label='CV Accuracy ± Std')
    ax.set_xscale('log')

    best_idx = int(np.argmax(mean_pct))
    ax.plot(C_values[best_idx], mean_pct[best_idx], 'g*',
            markersize=20, label=f'Best C={C_values[best_idx]}')

    ax.fill_between(C_values, mean_pct - std_pct, mean_pct + std_pct,
                     alpha=0.15, color='red')

    ax.set_xlabel('Regularization Parameter C')
    ax.set_ylabel('5-Fold CV Accuracy (%)')
    ax.set_title('Cross Validation — Accuracy vs C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  saved: {save_path}")


def print_summary(results):
    """Fig 5: Summary table."""
    print(f"\n{'='*80}")
    print(f"  {'Method':<28} {'Acc(%)':<10} {'F1(%)':<10} {'Train(s)':<12} {'Test(s)':<12}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['name']:<28} {r['accuracy']*100:<10.2f} {r['f1']*100:<10.2f} "
              f"{r['train_time']:<12.4f} {r['test_time']:<12.6f}")
    print(f"  {'='*72}")

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  Best accuracy: {best['name']} ({best['accuracy']*100:.2f}%)")
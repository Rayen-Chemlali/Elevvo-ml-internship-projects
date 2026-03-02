"""
final_comparison.py
Aggregates results from all approaches and produces a final comparison
bar chart and a printed summary.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import VIS_DIR, MODELS_DIR, RESULTS_DIR


def run_final_comparison(tabular_results, best_tabular_name, best_tabular_acc,
                         cnn_acc, transfer_acc):
    """
    Print a unified summary table and save a comparison bar chart.
    """
    print("\n" + "=" * 70)
    print("STEP 9: Final comparison of all approaches")
    print("=" * 70)

    rows = [
        {'Approach': f'Tabular — {best_tabular_name}',
         'Accuracy': best_tabular_acc, 'Type': 'Audio features'},
    ]
    if cnn_acc is not None:
        rows.append({'Approach': 'Custom CNN',
                     'Accuracy': cnn_acc, 'Type': 'Spectrograms'})
    if transfer_acc is not None:
        rows.append({'Approach': 'Transfer Learning (VGG16)',
                     'Accuracy': transfer_acc, 'Type': 'Spectrograms'})

    final_df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False)
    print("\n" + final_df.to_string(index=False))
    final_df.to_csv(os.path.join(RESULTS_DIR, 'final_comparison.csv'), index=False)

    # Bar chart
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(final_df['Approach'], final_df['Accuracy'],
                   color=colors[:len(final_df)])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Final comparison: Tabular vs CNN vs Transfer Learning', fontsize=14)
    plt.ylim(0, 1)
    for bar, acc in zip(bars, final_df['Accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'final_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved: final_comparison.png")

    _print_file_tree()

    print("\n" + "=" * 70)
    print("  TASK COMPLETE")
    print("=" * 70)


def _print_file_tree():
    print("\n" + "=" * 70)
    print("  GENERATED FILES")
    print("=" * 70)
    for folder in [VIS_DIR, MODELS_DIR, RESULTS_DIR]:
        print(f"\n  {os.path.basename(folder)}/")
        for root, dirs, files in os.walk(folder):
            level = root.replace(folder, '').count(os.sep)
            indent = '    ' + '  ' * level
            if level > 0:
                print(f'{indent}{os.path.basename(root)}/')
            for f in sorted(files):
                print(f'{indent}  {f}')

#!/usr/bin/env python
"""Generate paper-ready results and tables from experiment outputs."""

import pandas as pd
import numpy as np
from pathlib import Path

# Read all experiment results
results = {}
for exp_num in range(1, 6):
    try:
        df = pd.read_csv(f'outputs/tables/exp{exp_num}_results.csv')
        results[f'exp{exp_num}'] = df
        print(f"Loaded exp{exp_num}: {len(df)} rows")
    except Exception as e:
        print(f"Error loading exp{exp_num}: {e}")

print("\n" + "="*60)
print("EXPERIMENT 1: Label Shift with Offset Correction")
print("="*60)

exp1 = results['exp1']
# Summary by method and test prevalence
summary1 = exp1.groupby(['method', 'pi_test']).agg({
    'roc_auc': ['mean', 'std'],
    'risk': ['mean', 'std'],
    'precision': 'mean',
    'recall': 'mean'
}).round(3)

print("\nRisk by method and test prevalence (cost=1,1):")
risk_summary = exp1[exp1['c10'] == 1.0].pivot_table(
    values='risk', index='pi_test', columns='method', aggfunc='mean'
).round(3)
print(risk_summary)

print("\n" + "="*60)
print("EXPERIMENT 2: AUC Invariance")
print("="*60)

exp2 = results['exp2']
auc_by_pi = exp2.groupby('pi_test')['roc_auc'].agg(['mean', 'std']).round(4)
pr_auc_by_pi = exp2.groupby('pi_test')['pr_auc'].agg(['mean', 'std']).round(4)

print("\nAUC vs test prevalence:")
print(auc_by_pi)
print(f"\nAUC range: {auc_by_pi['mean'].max() - auc_by_pi['mean'].min():.4f}")

print("\nPR-AUC vs test prevalence:")
print(pr_auc_by_pi)

print("\n" + "="*60)
print("EXPERIMENT 3: Weighting and Effective Sample Size")
print("="*60)

exp3 = results['exp3']
neff_by_alpha = exp3.groupby('alpha')[['neff', 'roc_auc']].agg(['mean', 'std']).round(1)
print("\nEffective sample size by alpha:")
print(neff_by_alpha)

# Check angle stability if available
if 'mean_angle_deg' in exp3.columns:
    angle_by_alpha = exp3.groupby('alpha')['mean_angle_deg'].mean()
    print("\nMean coefficient angle (degrees) by alpha:")
    print(angle_by_alpha.round(2))

print("\n" + "="*60)
print("EXPERIMENT 4: Operating Point Metrics")
print("="*60)

exp4 = results['exp4']
op_metrics = exp4.groupby(['method', 'pi_test'])[['precision', 'recall']].mean().round(3)
print("\nPrecision and Recall by method and test prevalence:")
print(op_metrics)

print("\n" + "="*60)
print("EXPERIMENT 5: Concept Drift Control")
print("="*60)

exp5 = results['exp5']
drift_summary = exp5.groupby('method')[['roc_auc', 'risk']].agg(['mean', 'std']).round(4)
print("\nPerformance under concept drift:")
print(drift_summary)

# Calculate improvements
nocorr_auc = exp5[exp5['method'] == 'nocorr']['roc_auc'].mean()
offset_auc = exp5[exp5['method'] == 'offset']['roc_auc'].mean()
retrain_auc = exp5[exp5['method'] == 'retrain']['roc_auc'].mean()

print(f"\nOffset improvement: {offset_auc - nocorr_auc:.4f}")
print(f"Retrain improvement: {retrain_auc - nocorr_auc:.4f}")

# Create LaTeX table for paper
print("\n" + "="*60)
print("LATEX TABLE FOR PAPER")
print("="*60)

# Table 1: Main results summary
latex_data = []

# Exp 1 results at extreme prevalence
for pi in [0.5, 0.1, 0.01]:
    exp1_pi = exp1[(exp1['pi_test'] == pi) & (exp1['c10'] == 1.0)]
    row = {'$\\pi_{test}$': f'{pi:.2f}'}

    for method in ['nocorr', 'offset', 'oracle_threshold']:
        method_data = exp1_pi[exp1_pi['method'] == method]
        if len(method_data) > 0:
            risk = method_data['risk'].mean()
            risk_std = method_data['risk'].std()
            row[f'{method}_risk'] = f'{risk:.3f} $\\pm$ {risk_std:.3f}'

    latex_data.append(row)

latex_df = pd.DataFrame(latex_data)
print("\nTable 1: Risk under label shift")
print(latex_df.to_latex(index=False, escape=False))

# Save summary CSV
summary_all = pd.DataFrame({
    'Metric': ['AUC range (Exp1)', 'Neff @ α=50 (Exp3)', 'Concept drift retrain improvement (Exp5)'],
    'Value': [
        f"{auc_by_pi['mean'].max() - auc_by_pi['mean'].min():.4f}",
        f"{exp3[exp3['alpha'] == 50]['neff'].mean():.0f}" if 50 in exp3['alpha'].values else "N/A",
        f"{retrain_auc - nocorr_auc:.4f}"
    ]
})

summary_all.to_csv('outputs/tables/paper_summary.csv', index=False)
print("\nSaved paper summary to outputs/tables/paper_summary.csv")
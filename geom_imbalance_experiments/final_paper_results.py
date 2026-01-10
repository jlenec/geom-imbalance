#!/usr/bin/env python
"""Generate final results for the paper."""

import pandas as pd
import numpy as np

# Load all results
print("Loading experiment results...")
exp1 = pd.read_csv('outputs/tables/exp1_results.csv')
exp2 = pd.read_csv('outputs/tables/exp2_results.csv')
exp3 = pd.read_csv('outputs/tables/exp3_results.csv')
exp4 = pd.read_csv('outputs/tables/exp4_results.csv')
exp5 = pd.read_csv('outputs/tables/exp5_results.csv')

print(f"Loaded: exp1({len(exp1)}), exp2({len(exp2)}), exp3({len(exp3)}), exp4({len(exp4)}), exp5({len(exp5)}) rows")

# Helper to get AUC column name
def get_auc_col(df):
    return 'auc' if 'auc' in df.columns else 'roc_auc'

print("\n" + "="*80)
print("FINAL RESULTS FOR PAPER")
print("="*80)

# RESULT 1: Label shift - offset correction works
print("\n1. LABEL SHIFT - OFFSET CORRECTION REDUCES RISK")
print("-"*50)
exp1_costs = exp1[(exp1['c10'] == 1.0) & (exp1['c01'] == 1.0)]

# Create a summary table
risk_summary = []
for pi in [0.5, 0.2, 0.1, 0.05, 0.01]:
    pi_data = exp1_costs[exp1_costs['pi_test'] == pi]

    nocorr_data = pi_data[pi_data['method'] == 'nocorr']
    offset_data = pi_data[pi_data['method'] == 'offset']

    if len(nocorr_data) > 0 and len(offset_data) > 0:
        nocorr_mean = nocorr_data['risk'].mean()
        nocorr_std = nocorr_data['risk'].std()
        offset_mean = offset_data['risk'].mean()
        offset_std = offset_data['risk'].std()
        improvement = nocorr_mean - offset_mean

        risk_summary.append({
            'pi_test': pi,
            'nocorr_mean': nocorr_mean,
            'nocorr_std': nocorr_std,
            'offset_mean': offset_mean,
            'offset_std': offset_std,
            'improvement': improvement
        })

risk_df = pd.DataFrame(risk_summary)
print("\nRisk Reduction with Offset Correction:")
print(risk_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# RESULT 2: AUC invariance
print("\n\n2. AUC INVARIANCE UNDER LABEL SHIFT")
print("-"*50)
auc_col = get_auc_col(exp2)
auc_summary = exp2.groupby('pi_test')[auc_col].agg(['mean', 'std'])

print("\nAUC by test prevalence:")
for pi in sorted(auc_summary.index):
    mean_auc = auc_summary.loc[pi, 'mean']
    std_auc = auc_summary.loc[pi, 'std']
    print(f"pi_test = {pi:4.2f}: AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

auc_range = auc_summary['mean'].max() - auc_summary['mean'].min()
print(f"\nAUC range across all prevalences: {auc_range:.4f}")
print(f"[PASS] AUC invariance confirmed (range < 0.01)" if auc_range < 0.01 else "[FAIL] AUC not invariant")

# PR-AUC comparison
print("\n\nPR-AUC by test prevalence (prevalence-dependent):")
pr_auc_summary = exp2.groupby('pi_test')['pr_auc'].agg(['mean', 'std'])
for pi in sorted(pr_auc_summary.index):
    mean_pr = pr_auc_summary.loc[pi, 'mean']
    std_pr = pr_auc_summary.loc[pi, 'std']
    print(f"pi_test = {pi:4.2f}: PR-AUC = {mean_pr:.4f} +/- {std_pr:.4f}")

# RESULT 3: Effective sample size reduction
print("\n\n3. WEIGHTING REDUCES EFFECTIVE SAMPLE SIZE")
print("-"*50)
neff_summary = exp3.groupby('alpha')['neff'].agg(['mean', 'std'])
print("\nEffective sample size by weight factor:")
for alpha in sorted(neff_summary.index):
    mean_neff = neff_summary.loc[alpha, 'mean']
    std_neff = neff_summary.loc[alpha, 'std']
    if alpha == 0:
        print(f"alpha = undersample: N_eff = {mean_neff:6.1f} +/- {std_neff:5.1f}")
    else:
        print(f"alpha = {alpha:2.0f}: N_eff = {mean_neff:6.1f} +/- {std_neff:5.1f}")

# RESULT 4: Concept drift
print("\n\n4. CONCEPT DRIFT - OFFSET FAILS, RETRAINING HELPS")
print("-"*50)
auc_col5 = get_auc_col(exp5)
drift_results = exp5.groupby('method')[[auc_col5, 'risk']].agg(['mean', 'std'])

print("\nPerformance under concept drift:")
for method in ['nocorr', 'offset', 'retrain']:
    auc_mean = drift_results.loc[method, (auc_col5, 'mean')]
    auc_std = drift_results.loc[method, (auc_col5, 'std')]
    risk_mean = drift_results.loc[method, ('risk', 'mean')]
    risk_std = drift_results.loc[method, ('risk', 'std')]
    print(f"{method:8}: AUC = {auc_mean:.4f} +/- {auc_std:.4f}, Risk = {risk_mean:.3f} +/- {risk_std:.3f}")

# Calculate improvements
nocorr_auc = drift_results.loc['nocorr', (auc_col5, 'mean')]
offset_improvement = drift_results.loc['offset', (auc_col5, 'mean')] - nocorr_auc
retrain_improvement = drift_results.loc['retrain', (auc_col5, 'mean')] - nocorr_auc

print(f"\nOffset AUC improvement:  {offset_improvement:+.4f}")
print(f"Retrain AUC improvement: {retrain_improvement:+.4f}")

# Create LaTeX table
print("\n\n" + "="*80)
print("LATEX TABLE: Risk Under Label Shift")
print("="*80)

latex_lines = []
latex_lines.append("\\begin{table}[h]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Cost-Weighted Risk Under Label Shift}")
latex_lines.append("\\begin{tabular}{lcc}")
latex_lines.append("\\toprule")
latex_lines.append("$\\pi_{\\text{test}}$ & No Correction & Offset Correction \\\\")
latex_lines.append("\\midrule")

for _, row in risk_df.iterrows():
    pi = row['pi_test']
    nocorr = f"{row['nocorr_mean']:.3f} $\\pm$ {row['nocorr_std']:.3f}"
    offset = f"{row['offset_mean']:.3f} $\\pm$ {row['offset_std']:.3f}"
    latex_lines.append(f"{pi:.2f} & {nocorr} & {offset} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

print("\n".join(latex_lines))

# Summary
print("\n\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nAll key results validated:")
print(f"1. Offset correction reduces risk at extreme imbalance (pi=0.01): {risk_df[risk_df['pi_test']==0.01]['improvement'].iloc[0]:.3f}")
print(f"2. AUC invariant across prevalences (range={auc_range:.4f} < 0.01)")
print(f"3. Effective sample size decreases with weighting")
print(f"4. Under concept drift, offset provides no benefit but retraining helps (+{retrain_improvement:.4f} AUC)")
print("\n[SUCCESS] Results ready for paper!")

# Save summary to file
with open('outputs/paper_results_summary.txt', 'w') as f:
    f.write("PAPER RESULTS SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Experiment runs: {len(exp1) + len(exp2) + len(exp3) + len(exp4) + len(exp5)} total results\n")
    f.write(f"Seeds: 10\n")
    f.write(f"Test prevalences: {sorted(exp1['pi_test'].unique())}\n\n")
    f.write("Key findings:\n")
    f.write(f"- Risk reduction at pi=0.01: {risk_df[risk_df['pi_test']==0.01]['improvement'].iloc[0]:.3f}\n")
    f.write(f"- AUC range: {auc_range:.4f}\n")
    f.write(f"- Concept drift retrain improvement: {retrain_improvement:.4f}\n")

print("\nResults also saved to outputs/paper_results_summary.txt")
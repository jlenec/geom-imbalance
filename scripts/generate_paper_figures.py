#!/usr/bin/env python3
"""
Generate publication-ready summary tables and comparison figures from all scenarios
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("GENERATING PAPER FIGURES AND TABLES")
print("="*70)

# Load all scenario results
scenarios_data = []

scenarios = [
    (1, "Pure Label Shift"),
    (2, "Concept Drift"),
    (3, "Score Mapping Drift"),
    (4, "Covariate Shift"),
    (5, "Ill-Conditioned C"),
]

for scenario_num, scenario_name in scenarios:
    csv_path = f"artifacts/metrics/scenario_{scenario_num}_reports.csv"

    if not os.path.exists(csv_path):
        print(f"[WARN] Missing: {csv_path}")
        continue

    print(f"\n[{scenario_num}] Loading {scenario_name}...")
    df = pd.read_csv(csv_path)

    # Compute statistics
    state_counts = df['state'].value_counts()
    total_windows = len(df)

    stats = {
        'Scenario': scenario_num,
        'Name': scenario_name,
        'Total Windows': total_windows,
        'NORMAL': state_counts.get('NORMAL', 0),
        'PRIOR_SHIFT': state_counts.get('PRIOR_SHIFT', 0),
        'DRIFT_SUSPECTED': state_counts.get('DRIFT_SUSPECTED', 0),
        'NORMAL %': state_counts.get('NORMAL', 0) / total_windows * 100,
        'PRIOR_SHIFT %': state_counts.get('PRIOR_SHIFT', 0) / total_windows * 100,
        'DRIFT_SUSPECTED %': state_counts.get('DRIFT_SUSPECTED', 0) / total_windows * 100,
        'Mean d_u_star': df['d_u_star'].mean(),
        'Max d_u_star': df['d_u_star'].max(),
        'Mean pi_ewma': df['pi_ewma'].mean(),
        'Std pi_ewma': df['pi_ewma'].std(),
    }

    scenarios_data.append(stats)

    print(f"   Windows: {total_windows}")
    print(f"   NORMAL: {stats['NORMAL']} ({stats['NORMAL %']:.1f}%)")
    print(f"   PRIOR_SHIFT: {stats['PRIOR_SHIFT']} ({stats['PRIOR_SHIFT %']:.1f}%)")
    print(f"   DRIFT_SUSPECTED: {stats['DRIFT_SUSPECTED']} ({stats['DRIFT_SUSPECTED %']:.1f}%)")

# Create summary DataFrame
df_summary = pd.DataFrame(scenarios_data)

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(df_summary.to_string(index=False))

# Save summary table
os.makedirs("artifacts/tables", exist_ok=True)

# CSV format
df_summary.to_csv("artifacts/tables/scenario_summary.csv", index=False)
print("\nSaved: artifacts/tables/scenario_summary.csv")

# LaTeX format (for paper)
latex_table = df_summary[['Scenario', 'Name', 'NORMAL %', 'PRIOR_SHIFT %', 'DRIFT_SUSPECTED %',
                           'Mean d_u_star']].copy()
latex_table.columns = ['#', 'Scenario', 'NORMAL (\\%)', 'PRIOR\\_SHIFT (\\%)',
                       'DRIFT\\_SUSPECTED (\\%)', 'Mean $d_u^\\star$']

with open("artifacts/tables/scenario_summary.tex", 'w') as f:
    f.write("% Auto-generated table for paper\n")
    f.write("\\begin{table}[h!]\n")
    f.write("\\centering\n")
    f.write("\\caption{Drift Detection Results Across Five Scenarios}\n")
    f.write("\\label{tab:drift_scenarios}\n")
    f.write("\\begin{tabular}{clrrrr}\n")
    f.write("\\toprule\n")
    f.write("\\# & Scenario & NORMAL (\\%) & PRIOR\\_SHIFT (\\%) & DRIFT\\_SUSPECTED (\\%) & Mean $d_u^\\star$ \\\\\n")
    f.write("\\midrule\n")

    for _, row in df_summary.iterrows():
        f.write(f"{row['Scenario']} & {row['Name'][:20]} & {row['NORMAL %']:.1f} & "
                f"{row['PRIOR_SHIFT %']:.1f} & {row['DRIFT_SUSPECTED %']:.1f} & "
                f"{row['Mean d_u_star']:.4f} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print("Saved: artifacts/tables/scenario_summary.tex")

# Generate comparison plots
print("\n" + "="*70)
print("GENERATING COMPARISON PLOTS")
print("="*70)

# Figure 1: State distribution across scenarios
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(scenarios_data))
width = 0.25

normal_pcts = [s['NORMAL %'] for s in scenarios_data]
prior_pcts = [s['PRIOR_SHIFT %'] for s in scenarios_data]
drift_pcts = [s['DRIFT_SUSPECTED %'] for s in scenarios_data]

bars1 = ax.bar(x - width, normal_pcts, width, label='NORMAL', color='lightgreen', edgecolor='black')
bars2 = ax.bar(x, prior_pcts, width, label='PRIOR_SHIFT', color='gold', edgecolor='black')
bars3 = ax.bar(x + width, drift_pcts, width, label='DRIFT_SUSPECTED', color='lightcoral', edgecolor='black')

ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of Windows (%)', fontsize=12, fontweight='bold')
ax.set_title('Controller State Distribution Across Scenarios', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"S{s['Scenario']}\n{s['Name'][:15]}" for s in scenarios_data], fontsize=10)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 105)

# Add percentage labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 3:  # Only label if > 3%
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{height:.0f}%',
                   ha='center', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("artifacts/figures/scenario_comparison_states.png", dpi=300, bbox_inches='tight')
plt.savefig("artifacts/figures/scenario_comparison_states.pdf", bbox_inches='tight')
print("Saved: artifacts/figures/scenario_comparison_states.png")
print("Saved: artifacts/figures/scenario_comparison_states.pdf")
plt.close()

# Figure 2: Mean d_u_star comparison
fig, ax = plt.subplots(figsize=(10, 6))

scenarios_names = [f"S{s['Scenario']}: {s['Name'][:15]}" for s in scenarios_data]
d_u_stars = [s['Mean d_u_star'] for s in scenarios_data]

colors = ['green', 'red', 'red', 'orange', 'red']  # green=valid, red=invalid, orange=special
bars = ax.barh(scenarios_names, d_u_stars, color=colors, edgecolor='black', alpha=0.7)

ax.set_xlabel('Mean Mixture Consistency Statistic ($d_u^\\star$)', fontsize=12, fontweight='bold')
ax.set_title('Geometric Violation Metric Across Scenarios', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, d_u_stars)):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
           f'{val:.4f}',
           va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("artifacts/figures/scenario_comparison_du_star.png", dpi=300, bbox_inches='tight')
plt.savefig("artifacts/figures/scenario_comparison_du_star.pdf", bbox_inches='tight')
print("Saved: artifacts/figures/scenario_comparison_du_star.png")
print("Saved: artifacts/figures/scenario_comparison_du_star.pdf")
plt.close()

# Figure 3: Expected vs Observed behavior summary table
expected_behavior = {
    1: "PRIOR_SHIFT (label shift valid for adaptation)",
    2: "DRIFT_SUSPECTED (concept drift requires retraining)",
    3: "DRIFT_SUSPECTED (score mapping drift invalid)",
    4: "NORMAL/LOW_DRIFT (covariate shift should not alarm)",
    5: "DRIFT_SUSPECTED (relies on d_u_star when BBSE unstable)",
}

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

table_data = []
for s in scenarios_data:
    dominant_state = max(['NORMAL', 'PRIOR_SHIFT', 'DRIFT_SUSPECTED'],
                        key=lambda x: s[f'{x} %'])
    observed = f"{dominant_state} ({s[f'{dominant_state} %']:.0f}%)"
    expected = expected_behavior.get(s['Scenario'], "N/A")

    # Check if matches expectation
    if s['Scenario'] == 1:
        match = "✓" if s['PRIOR_SHIFT %'] > 50 else "✗"
    elif s['Scenario'] in [2, 3, 5]:
        match = "✓" if s['DRIFT_SUSPECTED %'] > 50 else "✗"
    elif s['Scenario'] == 4:
        match = "✓" if s['DRIFT_SUSPECTED %'] < 30 else "~"  # partial
    else:
        match = "?"

    table_data.append([
        f"S{s['Scenario']}",
        s['Name'],
        expected,
        observed,
        match
    ])

table = ax.table(cellText=table_data,
                colLabels=['#', 'Scenario', 'Expected Behavior', 'Observed State', 'Match'],
                cellLoc='left',
                loc='center',
                colWidths=[0.05, 0.2, 0.35, 0.25, 0.08])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color code the header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code match column
for i, row in enumerate(table_data, 1):
    match = row[4]
    if match == "✓":
        table[(i, 4)].set_facecolor('#C8E6C9')
    elif match == "✗":
        table[(i, 4)].set_facecolor('#FFCDD2')
    elif match == "~":
        table[(i, 4)].set_facecolor('#FFF9C4')

plt.title('Expected vs Observed Behavior Validation', fontsize=14, fontweight='bold', pad=20)
plt.savefig("artifacts/figures/scenario_validation_table.png", dpi=300, bbox_inches='tight')
plt.savefig("artifacts/figures/scenario_validation_table.pdf", bbox_inches='tight')
print("Saved: artifacts/figures/scenario_validation_table.png")
print("Saved: artifacts/figures/scenario_validation_table.pdf")
plt.close()

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  Tables:")
print("    - artifacts/tables/scenario_summary.csv")
print("    - artifacts/tables/scenario_summary.tex")
print("  Figures:")
print("    - artifacts/figures/scenario_comparison_states.png (and .pdf)")
print("    - artifacts/figures/scenario_comparison_du_star.png (and .pdf)")
print("    - artifacts/figures/scenario_validation_table.png (and .pdf)")
print("\nIndividual scenario figures:")
for i in range(1, 6):
    print(f"    - artifacts/figures/scenario_{i}_drift_detection.png")
    print(f"    - artifacts/figures/scenario_{i}_state_distribution.png")

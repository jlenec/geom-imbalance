import pandas as pd

# Check exp1 results
exp1 = pd.read_csv('outputs/tables/exp1_results.csv')
print("Experiment 1 AUC by prevalence:")
auc_summary = exp1.groupby(['dataset', 'model', 'pi_test'])['roc_auc'].agg(['mean', 'std', 'count'])
print(auc_summary)
print("\nAUC range check:")
for (dataset, model), group in exp1.groupby(['dataset', 'model']):
    auc_by_pi = group.groupby('pi_test')['roc_auc'].mean()
    auc_range = auc_by_pi.max() - auc_by_pi.min()
    print(f"{dataset} - {model}: range = {auc_range:.4f}")

# Check exp3 results
print("\n\nExperiment 3 Neff by alpha:")
exp3 = pd.read_csv('outputs/tables/exp3_results.csv')
neff_summary = exp3.groupby('alpha')['neff'].agg(['mean', 'count'])
print(neff_summary)

# Check exp5 results
print("\n\nExperiment 5 Concept Drift:")
exp5 = pd.read_csv('outputs/tables/exp5_results.csv')
perf_summary = exp5.groupby('method')['roc_auc'].agg(['mean', 'std'])
print(perf_summary)
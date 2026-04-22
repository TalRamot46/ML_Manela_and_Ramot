import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_data = """
Running inner cross-validation for outer fold 1/5 with 25 models M_i...
Running M_1 with params {'C': 1.0, 'gamma': 0.0001}...  finished in 24.69 seconds | mean inner error = 0.2095
Running M_2 with params {'C': 1.0, 'gamma': 0.01}...    finished in 38.74 seconds | mean inner error = 0.1528
Running M_3 with params {'C': 1.0, 'gamma': 1}...       finished in 46.20 seconds | mean inner error = 0.9634
Running M_4 with params {'C': 1.0, 'gamma': 100}...     finished in 39.39 seconds | mean inner error = 0.9634
Running M_5 with params {'C': 1.0, 'gamma': 'scale'}... finished in 23.86 seconds | mean inner error = 0.0636
Running M_6 with params {'C': 10.0, 'gamma': 0.0001}... finished in 17.54 seconds | mean inner error = 0.1318
Running M_7 with params {'C': 10.0, 'gamma': 0.01}...   finished in 35.61 seconds | mean inner error = 0.1476
Running M_8 with params {'C': 10.0, 'gamma': 1}...      finished in 51.81 seconds | mean inner error = 0.9634
Running M_9 with params {'C': 10.0, 'gamma': 100}...    finished in 42.64 seconds | mean inner error = 0.9634
Running M_10 with params {'C': 10.0, 'gamma': 'scale'}...       finished in 22.74 seconds | mean inner error = 0.0482
Running M_11 with params {'C': 100.0, 'gamma': 0.0001}...       finished in 15.64 seconds | mean inner error = 0.1005
Running M_12 with params {'C': 100.0, 'gamma': 0.01}... finished in 38.10 seconds | mean inner error = 0.1476
Running M_13 with params {'C': 100.0, 'gamma': 1}...    finished in 51.25 seconds | mean inner error = 0.9634
Running M_14 with params {'C': 100.0, 'gamma': 100}...  finished in 37.63 seconds | mean inner error = 0.9634
Running M_15 with params {'C': 100.0, 'gamma': 'scale'}...      finished in 24.21 seconds | mean inner error = 0.0482
Running M_16 with params {'C': 1000.0, 'gamma': 0.0001}...      finished in 16.97 seconds | mean inner error = 0.1004
Running M_17 with params {'C': 1000.0, 'gamma': 0.01}...        finished in 39.88 seconds | mean inner error = 0.1476
Running M_18 with params {'C': 1000.0, 'gamma': 1}...   finished in 49.74 seconds | mean inner error = 0.9634
Running M_19 with params {'C': 1000.0, 'gamma': 100}... finished in 41.68 seconds | mean inner error = 0.9634
Running M_20 with params {'C': 1000.0, 'gamma': 'scale'}...     finished in 20.69 seconds | mean inner error = 0.0482
Running M_21 with params {'C': 10000.0, 'gamma': 0.0001}...     finished in 17.28 seconds | mean inner error = 0.1005
Running M_22 with params {'C': 10000.0, 'gamma': 0.01}...       finished in 37.61 seconds | mean inner error = 0.1476
Running M_23 with params {'C': 10000.0, 'gamma': 1}...  finished in 49.66 seconds | mean inner error = 0.9634
Running M_24 with params {'C': 10000.0, 'gamma': 100}...        finished in 38.35 seconds | mean inner error = 0.9634
Running M_25 with params {'C': 10000.0, 'gamma': 'scale'}...    finished in 23.12 seconds | mean inner error = 0.0482

Running inner cross-validation for outer fold 2/5 with 25 models M_i...
Running M_1 with params {'C': 1.0, 'gamma': 0.0001}...  finished in 26.74 seconds | mean inner error = 0.2079
Running M_2 with params {'C': 1.0, 'gamma': 0.01}...    finished in 30.77 seconds | mean inner error = 0.1566
Running M_3 with params {'C': 1.0, 'gamma': 1}...       finished in 48.10 seconds | mean inner error = 0.9633
Running M_4 with params {'C': 1.0, 'gamma': 100}...     finished in 34.31 seconds | mean inner error = 0.9633
Running M_5 with params {'C': 1.0, 'gamma': 'scale'}... finished in 19.65 seconds | mean inner error = 0.0658
Running M_6 with params {'C': 10.0, 'gamma': 0.0001}... finished in 12.84 seconds | mean inner error = 0.1311
Running M_7 with params {'C': 10.0, 'gamma': 0.01}...   finished in 23.70 seconds | mean inner error = 0.1507
Running M_8 with params {'C': 10.0, 'gamma': 1}...      finished in 28.03 seconds | mean inner error = 0.9633
Running M_9 with params {'C': 10.0, 'gamma': 100}...    finished in 23.85 seconds | mean inner error = 0.9633
Running M_10 with params {'C': 10.0, 'gamma': 'scale'}...       finished in 14.50 seconds | mean inner error = 0.0522
Running M_11 with params {'C': 100.0, 'gamma': 0.0001}...       finished in 11.32 seconds | mean inner error = 0.0994
Running M_12 with params {'C': 100.0, 'gamma': 0.01}... finished in 22.63 seconds | mean inner error = 0.1507
Running M_13 with params {'C': 100.0, 'gamma': 1}...    finished in 35.34 seconds | mean inner error = 0.9633
Running M_14 with params {'C': 100.0, 'gamma': 100}...  finished in 33.17 seconds | mean inner error = 0.9633
Running M_15 with params {'C': 100.0, 'gamma': 'scale'}...      finished in 24.70 seconds | mean inner error = 0.0521
Running M_16 with params {'C': 1000.0, 'gamma': 0.0001}...      finished in 16.16 seconds | mean inner error = 0.0984
Running M_17 with params {'C': 1000.0, 'gamma': 0.01}...        finished in 52.99 seconds | mean inner error = 0.1507
Running M_18 with params {'C': 1000.0, 'gamma': 1}...   finished in 51.49 seconds | mean inner error = 0.9633
Running M_19 with params {'C': 1000.0, 'gamma': 100}... finished in 50.01 seconds | mean inner error = 0.9633
Running M_20 with params {'C': 1000.0, 'gamma': 'scale'}...     finished in 38.15 seconds | mean inner error = 0.0521
Running M_21 with params {'C': 10000.0, 'gamma': 0.0001}...     finished in 17.15 seconds | mean inner error = 0.0985
Running M_22 with params {'C': 10000.0, 'gamma': 0.01}...       finished in 25.85 seconds | mean inner error = 0.1507
Running M_23 with params {'C': 10000.0, 'gamma': 1}...  finished in 34.34 seconds | mean inner error = 0.9633
Running M_24 with params {'C': 10000.0, 'gamma': 100}...        finished in 29.61 seconds | mean inner error = 0.9633
Running M_25 with params {'C': 10000.0, 'gamma': 'scale'}...    finished in 20.77 seconds | mean inner error = 0.0521

Running inner cross-validation for outer fold 3/5 with 25 models M_i...
Running M_1 with params {'C': 1.0, 'gamma': 0.0001}...  finished in 19.49 seconds | mean inner error = 0.2117
Running M_2 with params {'C': 1.0, 'gamma': 0.01}...    finished in 30.64 seconds | mean inner error = 0.1536
Running M_3 with params {'C': 1.0, 'gamma': 1}...       finished in 33.54 seconds | mean inner error = 0.9633
Running M_4 with params {'C': 1.0, 'gamma': 100}...     finished in 25.25 seconds | mean inner error = 0.9633
Running M_5 with params {'C': 1.0, 'gamma': 'scale'}... finished in 15.60 seconds | mean inner error = 0.0694
Running M_6 with params {'C': 10.0, 'gamma': 0.0001}... finished in 14.33 seconds | mean inner error = 0.1326
Running M_7 with params {'C': 10.0, 'gamma': 0.01}...   finished in 26.84 seconds | mean inner error = 0.1483
Running M_8 with params {'C': 10.0, 'gamma': 1}...      finished in 27.30 seconds | mean inner error = 0.9633
Running M_9 with params {'C': 10.0, 'gamma': 100}...    finished in 27.25 seconds | mean inner error = 0.9633
Running M_10 with params {'C': 10.0, 'gamma': 'scale'}...       finished in 17.22 seconds | mean inner error = 0.0515
Running M_11 with params {'C': 100.0, 'gamma': 0.0001}...       finished in 15.41 seconds | mean inner error = 0.1038
Running M_12 with params {'C': 100.0, 'gamma': 0.01}... finished in 32.25 seconds | mean inner error = 0.1483
Running M_13 with params {'C': 100.0, 'gamma': 1}...    finished in 28.48 seconds | mean inner error = 0.9633
Running M_14 with params {'C': 100.0, 'gamma': 100}...  finished in 33.22 seconds | mean inner error = 0.9633
Running M_15 with params {'C': 100.0, 'gamma': 'scale'}...      finished in 23.29 seconds | mean inner error = 0.0515
Running M_16 with params {'C': 1000.0, 'gamma': 0.0001}...      finished in 18.17 seconds | mean inner error = 0.1014
Running M_17 with params {'C': 1000.0, 'gamma': 0.01}...        finished in 26.91 seconds | mean inner error = 0.1483
Running M_18 with params {'C': 1000.0, 'gamma': 1}...   finished in 25.86 seconds | mean inner error = 0.9633
Running M_19 with params {'C': 1000.0, 'gamma': 100}... finished in 31.12 seconds | mean inner error = 0.9633
Running M_20 with params {'C': 1000.0, 'gamma': 'scale'}...     finished in 17.69 seconds | mean inner error = 0.0515
Running M_21 with params {'C': 10000.0, 'gamma': 0.0001}...     finished in 14.14 seconds | mean inner error = 0.1015
Running M_22 with params {'C': 10000.0, 'gamma': 0.01}...       finished in 27.64 seconds | mean inner error = 0.1483
Running M_23 with params {'C': 10000.0, 'gamma': 1}...  finished in 32.90 seconds | mean inner error = 0.9633
Running M_24 with params {'C': 10000.0, 'gamma': 100}...        finished in 31.48 seconds | mean inner error = 0.9633
Running M_25 with params {'C': 10000.0, 'gamma': 'scale'}...    finished in 18.49 seconds | mean inner error = 0.0515

Running inner cross-validation for outer fold 4/5 with 25 models M_i...
Running M_1 with params {'C': 1.0, 'gamma': 0.0001}...  finished in 19.49 seconds | mean inner error = 0.2047
Running M_2 with params {'C': 1.0, 'gamma': 0.01}...    finished in 26.19 seconds | mean inner error = 0.1475
Running M_3 with params {'C': 1.0, 'gamma': 1}...       finished in 42.74 seconds | mean inner error = 0.9633
Running M_4 with params {'C': 1.0, 'gamma': 100}...     finished in 38.18 seconds | mean inner error = 0.9633
Running M_5 with params {'C': 1.0, 'gamma': 'scale'}... finished in 19.79 seconds | mean inner error = 0.0642
Running M_6 with params {'C': 10.0, 'gamma': 0.0001}... finished in 15.25 seconds | mean inner error = 0.1307
Running M_7 with params {'C': 10.0, 'gamma': 0.01}...   finished in 28.67 seconds | mean inner error = 0.1411
Running M_8 with params {'C': 10.0, 'gamma': 1}...      finished in 31.45 seconds | mean inner error = 0.9633
Running M_9 with params {'C': 10.0, 'gamma': 100}...    finished in 28.71 seconds | mean inner error = 0.9633
Running M_10 with params {'C': 10.0, 'gamma': 'scale'}...       finished in 17.95 seconds | mean inner error = 0.0505
Running M_11 with params {'C': 100.0, 'gamma': 0.0001}...       finished in 13.83 seconds | mean inner error = 0.1008
Running M_12 with params {'C': 100.0, 'gamma': 0.01}... finished in 32.57 seconds | mean inner error = 0.1411
Running M_13 with params {'C': 100.0, 'gamma': 1}...    finished in 37.19 seconds | mean inner error = 0.9633
Running M_14 with params {'C': 100.0, 'gamma': 100}...  finished in 31.04 seconds | mean inner error = 0.9633
Running M_15 with params {'C': 100.0, 'gamma': 'scale'}...      finished in 18.45 seconds | mean inner error = 0.0505
Running M_16 with params {'C': 1000.0, 'gamma': 0.0001}...      finished in 13.56 seconds | mean inner error = 0.0994
Running M_17 with params {'C': 1000.0, 'gamma': 0.01}...        finished in 24.66 seconds | mean inner error = 0.1411
Running M_18 with params {'C': 1000.0, 'gamma': 1}...   finished in 36.20 seconds | mean inner error = 0.9633
Running M_19 with params {'C': 1000.0, 'gamma': 100}... finished in 33.23 seconds | mean inner error = 0.9633
Running M_20 with params {'C': 1000.0, 'gamma': 'scale'}...     finished in 19.56 seconds | mean inner error = 0.0505
Running M_21 with params {'C': 10000.0, 'gamma': 0.0001}...     finished in 12.16 seconds | mean inner error = 0.0994
Running M_22 with params {'C': 10000.0, 'gamma': 0.01}...       finished in 27.61 seconds | mean inner error = 0.1411
Running M_23 with params {'C': 10000.0, 'gamma': 1}...  finished in 33.30 seconds | mean inner error = 0.9633
Running M_24 with params {'C': 10000.0, 'gamma': 100}...        finished in 30.77 seconds | mean inner error = 0.9633
Running M_25 with params {'C': 10000.0, 'gamma': 'scale'}...    finished in 19.26 seconds | mean inner error = 0.0505

Running inner cross-validation for outer fold 5/5 with 25 models M_i...
Running M_1 with params {'C': 1.0, 'gamma': 0.0001}...  finished in 21.36 seconds | mean inner error = 0.2084
Running M_2 with params {'C': 1.0, 'gamma': 0.01}...    finished in 30.96 seconds | mean inner error = 0.1494
Running M_3 with params {'C': 1.0, 'gamma': 1}...       finished in 36.54 seconds | mean inner error = 0.9626
Running M_4 with params {'C': 1.0, 'gamma': 100}...     finished in 28.11 seconds | mean inner error = 0.9634
Running M_5 with params {'C': 1.0, 'gamma': 'scale'}... finished in 17.74 seconds | mean inner error = 0.0689
Running M_6 with params {'C': 10.0, 'gamma': 0.0001}... finished in 15.44 seconds | mean inner error = 0.1346
Running M_7 with params {'C': 10.0, 'gamma': 0.01}...   finished in 30.25 seconds | mean inner error = 0.1444
Running M_8 with params {'C': 10.0, 'gamma': 1}...      finished in 37.79 seconds | mean inner error = 0.9616
Running M_9 with params {'C': 10.0, 'gamma': 100}...    finished in 34.18 seconds | mean inner error = 0.9634
Running M_10 with params {'C': 10.0, 'gamma': 'scale'}...       finished in 29.31 seconds | mean inner error = 0.0514
Running M_11 with params {'C': 100.0, 'gamma': 0.0001}...       finished in 24.70 seconds | mean inner error = 0.1029
Running M_12 with params {'C': 100.0, 'gamma': 0.01}... finished in 46.38 seconds | mean inner error = 0.1444
Running M_13 with params {'C': 100.0, 'gamma': 1}...    finished in 43.14 seconds | mean inner error = 0.9616
Running M_14 with params {'C': 100.0, 'gamma': 100}...  finished in 42.36 seconds | mean inner error = 0.9634
Running M_15 with params {'C': 100.0, 'gamma': 'scale'}...      finished in 26.49 seconds | mean inner error = 0.0514
Running M_16 with params {'C': 1000.0, 'gamma': 0.0001}...      finished in 21.78 seconds | mean inner error = 0.1004
Running M_17 with params {'C': 1000.0, 'gamma': 0.01}...        finished in 41.23 seconds | mean inner error = 0.1444
Running M_18 with params {'C': 1000.0, 'gamma': 1}...   finished in 51.40 seconds | mean inner error = 0.9616
Running M_19 with params {'C': 1000.0, 'gamma': 100}... finished in 38.38 seconds | mean inner error = 0.9634
Running M_20 with params {'C': 1000.0, 'gamma': 'scale'}...     finished in 23.92 seconds | mean inner error = 0.0514
Running M_21 with params {'C': 10000.0, 'gamma': 0.0001}...     finished in 18.30 seconds | mean inner error = 0.1004
Running M_22 with params {'C': 10000.0, 'gamma': 0.01}...       finished in 39.25 seconds | mean inner error = 0.1444
Running M_23 with params {'C': 10000.0, 'gamma': 1}...  finished in 53.51 seconds | mean inner error = 0.9616
Running M_24 with params {'C': 10000.0, 'gamma': 100}...        finished in 37.60 seconds | mean inner error = 0.9634
Running M_25 with params {'C': 10000.0, 'gamma': 'scale'}...    finished in 21.62 seconds | mean inner error = 0.0514
"""

results = []
current_fold = 0
pattern = re.compile(r"Running M_(\d+) with params (\{.*\}).*mean inner error = ([\d\.]+)")

for line in log_data.split('\n'):
    if "outer fold" in line:
        current_fold = int(re.search(r"fold (\d)/5", line).group(1))
    match = pattern.search(line)
    if match:
        params = eval(match.group(2))
        error = float(match.group(3))
        results.append({
            'fold': current_fold,
            'C': params['C'],
            'gamma': str(params['gamma']), # Keep as string for now to handle 'scale'
            'error': error
        })

df = pd.DataFrame(results)

# Average across folds
avg_df = df.groupby(['C', 'gamma']).agg({'error': 'mean'}).reset_index()

# For line plots, let's create a numerical gamma for sorting
def gamma_val(g):
    if g == 'scale': return 0.001137
    return float(g)

avg_df['gamma_num'] = avg_df['gamma'].apply(gamma_val)
avg_df = avg_df.sort_values(['C', 'gamma_num'])

# Plot 1: Error vs C (Log scale)
plt.figure(figsize=(10, 6))
# Unique gammas for labels
gamma_order = sorted(avg_df['gamma'].unique(), key=gamma_val)
sns.lineplot(data=avg_df, x='C', y='error', hue='gamma', hue_order=gamma_order, marker='o')
plt.xscale('log')
plt.title('SVM: Mean Inner Error vs C')
plt.ylabel('Mean Inner Error')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('svm_error_vs_c.png')
plt.close()

# Plot 2: Error vs Gamma
plt.figure(figsize=(10, 6))
# We will plot on categorical axis to handle 'scale' string nicely
sns.lineplot(data=avg_df, x='gamma_num', y='error', hue='C', palette='viridis', marker='o')
plt.xscale('log')
plt.title('SVM: Mean Inner Error vs Gamma')
plt.xlabel('Gamma (theoretical scale ~0.0011)')
plt.ylabel('Mean Inner Error')
plt.grid(True, which="both", ls="-", alpha=0.5)
# Set x-ticks to show actual gamma labels
plt.xticks(avg_df['gamma_num'].unique(), avg_df['gamma'].unique())
plt.savefig('svm_error_vs_gamma.png')
plt.close()

# Plot 3: Heatmap (C vs Gamma)
plt.figure(figsize=(10, 6))
# Re-pivot with categorical gamma
heatmap_data = avg_df.pivot(index='C', columns='gamma', values='error')
# Order columns by gamma_num
heatmap_data = heatmap_data[gamma_order]
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='RdYlGn_r')
plt.title('Heatmap: SVM Mean Inner Error (C vs Gamma)')
plt.savefig('svm_heatmap.png')
plt.close()

print(avg_df.head())
avg_df.to_csv('svm_average_inner_errors.csv', index=False)
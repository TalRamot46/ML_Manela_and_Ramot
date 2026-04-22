import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

log_text = """
Running inner cross-validation for outer fold 1/5 with 4 models M_i...
Running M_1 with params {'max_depth': 2, 'min_samples_split': 2}...     finished in 231.81 seconds | mean inner error = 0.9029
Running M_2 with params {'max_depth': 2, 'min_samples_split': 4}...     finished in 208.83 seconds | mean inner error = 0.9029
Running M_3 with params {'max_depth': 4, 'min_samples_split': 2}...     finished in 372.85 seconds | mean inner error = 0.8526
Running M_4 with params {'max_depth': 4, 'min_samples_split': 4}...     finished in 379.90 seconds | mean inner error = 0.8526
Running inner cross-validation for outer fold 2/5 with 4 models M_i...
Running M_1 with params {'max_depth': 2, 'min_samples_split': 2}...     finished in 285.18 seconds | mean inner error = 0.9093
Running M_2 with params {'max_depth': 2, 'min_samples_split': 4}...     finished in 232.37 seconds | mean inner error = 0.9093
Running M_3 with params {'max_depth': 4, 'min_samples_split': 2}...     finished in 516.71 seconds | mean inner error = 0.8540
Running M_4 with params {'max_depth': 4, 'min_samples_split': 4}...     finished in 392.90 seconds | mean inner error = 0.8540
Running inner cross-validation for outer fold 3/5 with 4 models M_i...
Running M_1 with params {'max_depth': 2, 'min_samples_split': 2}...     finished in 168.29 seconds | mean inner error = 0.9031
Running M_2 with params {'max_depth': 2, 'min_samples_split': 4}...     finished in 172.35 seconds | mean inner error = 0.9031
Running M_3 with params {'max_depth': 4, 'min_samples_split': 2}...     finished in 265.21 seconds | mean inner error = 0.8525
Running M_4 with params {'max_depth': 4, 'min_samples_split': 4}...     finished in 276.81 seconds | mean inner error = 0.8525
Running inner cross-validation for outer fold 4/5 with 4 models M_i...
Running M_1 with params {'max_depth': 2, 'min_samples_split': 2}...     finished in 168.99 seconds | mean inner error = 0.9050
Running M_2 with params {'max_depth': 2, 'min_samples_split': 4}...     finished in 173.18 seconds | mean inner error = 0.9050
Running M_3 with params {'max_depth': 4, 'min_samples_split': 2}...     finished in 296.76 seconds | mean inner error = 0.8462
Running M_4 with params {'max_depth': 4, 'min_samples_split': 4}...     finished in 280.22 seconds | mean inner error = 0.8462
Running inner cross-validation for outer fold 5/5 with 4 models M_i...
Running M_1 with params {'max_depth': 2, 'min_samples_split': 2}...     finished in 166.50 seconds | mean inner error = 0.9083
Running M_2 with params {'max_depth': 2, 'min_samples_split': 4}...     finished in 157.76 seconds | mean inner error = 0.9083
Running M_3 with params {'max_depth': 4, 'min_samples_split': 2}...     finished in 399.09 seconds | mean inner error = 0.8518
Running M_4 with params {'max_depth': 4, 'min_samples_split': 4}...     finished in 414.27 seconds | mean inner error = 0.8518
"""

import re

data = []
current_fold = 0
fold_pattern = re.compile(r"outer fold (\d)/5")
model_pattern = re.compile(r"Running M_(\d) with params \{'max_depth': (\d), 'min_samples_split': (\d)\}.*mean inner error = ([\d\.]+)")

for line in log_text.strip().split('\n'):
    fold_match = fold_pattern.search(line)
    if fold_match:
        current_fold = int(fold_match.group(1))
    
    model_match = model_pattern.search(line)
    if model_match:
        m_id = int(model_match.group(1))
        max_depth = int(model_match.group(2))
        min_split = int(model_match.group(3))
        inner_error = float(model_match.group(4))
        data.append({
            'outer_fold': current_fold,
            'max_depth': max_depth,
            'min_samples_split': min_split,
            'inner_error': inner_error
        })

df = pd.DataFrame(data)

# Calculate averages
avg_inner_errors = df.groupby(['max_depth', 'min_samples_split'])['inner_error'].mean().reset_index()

# Plotting Heatmap
plt.figure(figsize=(8, 6))
heatmap_data = avg_inner_errors.pivot(index='max_depth', columns='min_samples_split', values='inner_error')
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='YlOrRd')
plt.title('Mean Inner Error: Max Depth vs Min Samples Split')
plt.savefig('tree_heatmap.png')
plt.close()

# Plotting Error per Fold (Line Plot)
plt.figure(figsize=(10, 6))
df['params'] = df.apply(lambda x: f"D={int(x['max_depth'])}, S={int(x['min_samples_split'])}", axis=1)
sns.lineplot(data=df, x='outer_fold', y='inner_error', hue='params', marker='o')
plt.title('Inner Error per Outer Fold for each Hyperparameter Set')
plt.xlabel('Outer Fold')
plt.ylabel('Mean Inner Error')
plt.xticks([1,2,3,4,5])
plt.legend(title='Hyperparameters')
plt.grid(True)
plt.savefig('tree_error_per_fold.png')
plt.close()

print(avg_inner_errors)
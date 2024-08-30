import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv('/Users/bytedance/Desktop/code/Vision_Feature_AC_private-master/visualization/ablations_t.csv')

# Define the parameters
benchmarks = ["mmbench_en", "mme", "mmmu_val", "ok_vqa", "textvqa_val", "vizwiz_vqa_val", "scienceqa_img", "seed_image"]
modes = ["average"]
train_models = ["CLIP224", "OpenCLIP", "DINOv2", "SDim", "SD1.5", "SDXL", "DiT", "SD3", "SD2.1","CLIP336", "SigLIP", "CLIP224+DINOv2", "CLIP336+DINOv2"]
test_models = ["CLIP336+DINOv2"]
corres = ['corres']

# Initialize a dictionary to store the results
results = {'Benchmarks': benchmarks}
predicted_scores = {benchmark: {} for benchmark in benchmarks}

column_names = df.columns.tolist()

# Loop through the combinations and compute the average error rates
for mode in modes:
    for c in corres:
        for benchmark in benchmarks:
            data_perf_max = df[benchmark].min()
            data_perf_min = df[benchmark].max()
            data_a_max = df[f'{benchmark}_{mode}'].max()
            data_a_min = df[f'{benchmark}_{mode}'].min()
            data_c_max = df[c].max()
            data_c_min = df[c].min()

            df["normed_a"] = (df[f'{benchmark}_{mode}'] - data_a_min) / (data_a_max - data_a_min)
            df["normed_c"] = (df[c] - data_c_min) / (data_c_max - data_c_min)

            train_df = df[df['model'].isin(train_models)]
            test_df = df[df['model'].isin(test_models)]

            normalized_train_y = (train_df[benchmark] - data_perf_min) / (data_perf_max - data_perf_min)
            normalized_train_X = train_df[['normed_a', 'normed_c']]
            normalized_test_y = (test_df[benchmark] - data_perf_min) / (data_perf_max - data_perf_min)
            normalized_test_X = test_df[['normed_a', 'normed_c']]

            # Polynomial regression (degree 2)
            poly = PolynomialFeatures(degree=2)
            train_X_poly = poly.fit_transform(normalized_train_X)
            poly_model = LinearRegression()
            poly_model.fit(train_X_poly, normalized_train_y)
            train_pred = poly_model.predict(train_X_poly)

            train_mse = mean_squared_error(normalized_train_y, train_pred)

            test_X_poly = poly.transform(normalized_test_X)
            test_pred = poly_model.predict(test_X_poly)
            test_mse = mean_squared_error(normalized_test_y, test_pred)

            # Store predicted scores
            for index, row in test_df.iterrows():
                model_name = row['model']
                normalized_test_X_single = poly.transform([[row['normed_a'], row['normed_c']]])
                predicted_score = poly_model.predict(normalized_test_X_single)[0]
                if model_name not in predicted_scores[benchmark]:
                    predicted_scores[benchmark][model_name] = predicted_score
                else:
                    predicted_scores[benchmark][model_name] = max(predicted_scores[benchmark][model_name], predicted_score)

# Print the model with the highest predicted score for each benchmark
for benchmark, scores in predicted_scores.items():
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    print(f"Benchmark: {benchmark}, Best Model: {best_model}, Predicted Score: {best_score}")

# Radar plot setup
categories = benchmarks
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Normalize the predicted scores for radar plot
normalized_scores = {}
for benchmark in benchmarks:
    max_score = max(predicted_scores[benchmark].values())
    min_score = min(predicted_scores[benchmark].values())
    normalized_scores[benchmark] = {model: (score - min_score) / (max_score - min_score) for model, score in predicted_scores[benchmark].items()}

# Plot each model's performance
for model in test_models:
    values = [normalized_scores[benchmark][model] for benchmark in benchmarks]
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
    ax.fill(angles, values, alpha=0.25)

# Customize the radar chart
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Shrink the radar chart circle by adjusting the subplot parameters
fig.subplots_adjust(left=0.25, right=0.75, top=0.75, bottom=0.25)

# Add labels for each benchmark
for i, angle in enumerate(angles[:-1]):
    category = categories[i]
    x = np.cos(angle) * 1.2
    y = np.sin(angle) * 1.2
    ax.text(x, y, category, horizontalalignment='center', size=12, color='black', weight='semibold')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('Model Performance Comparison with Custom Scales')
plt.show()

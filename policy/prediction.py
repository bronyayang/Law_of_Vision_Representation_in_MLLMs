import pandas as pd
import itertools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv('/Users/shijiayang/Desktop/Vision_Feature_AC_private/visualizations/ablations_t.csv')

# Define the parameters
benchmarks = ["mmbench_en", "mme", "mmmu_val", "ok_vqa", "textvqa_val", "vizwiz_vqa_val", "scienceqa_img", "seed_image"]
optimal = ["CLIP224+DINOv2", "CLIP336", "OpenCLIP", "CLIP336+DINOv2", "CLIP336+DINOv2", "CLIP336", "CLIP336", "CLIP336+DINOv2"]

# Define the possible models
all_models = ["CLIP336", "CLIP224", "OpenCLIP", "DINOv2", "SDim", "SD1.5", "SDXL", "DiT", "SD3", "SD2.1", "SigLIP", "CLIP224+DINOv2", "CLIP336+DINOv2"]

found_optimal = [float("inf") for _ in range(8)]
length_optimal = [float("inf") for _ in range(8)]

results = []

# Loop through the combinations and compute the average error rates
for train_model_count in range(2, len(all_models) + 1):
    for train_models in itertools.combinations(all_models, train_model_count):
        train_models_set = set(train_models)
        test_models = [model for model in all_models if model not in train_models_set]
        
        for i, benchmark in enumerate(benchmarks):
            # if optimal[i] in train_models:
            #     continue
            
            # if length_optimal[i] + 1 < train_model_count and found_optimal[i] == 0:
            #     continue

            data_perf_max = df[benchmark].max()
            data_perf_min = df[benchmark].min()
            data_a_max = df[f'{benchmark}_average'].max()
            data_a_min = df[f'{benchmark}_average'].min()
            data_c_max = df['corres'].max()
            data_c_min = df['corres'].min()

            df["normed_a"] = (df[f'{benchmark}_average'] - data_a_min) / (data_a_max - data_a_min)
            df["normed_c"] = (df['corres'] - data_c_min) / (data_c_max - data_c_min)

            train_df = df[df['model'].isin(train_models)]
            test_df = df[df['model'].isin(test_models)]

            if train_df.empty or test_df.empty:
                continue

            normalized_train_y = (train_df[benchmark] - data_perf_min) / (data_perf_max - data_perf_min)
            normalized_train_X = train_df[['normed_a', 'normed_c']]
            normalized_test_y = (test_df[benchmark] - data_perf_min) / (data_perf_max - data_perf_min)
            normalized_test_X = test_df[['normed_a', 'normed_c']]

            poly = PolynomialFeatures(degree=2)
            normalized_train_X = poly.fit_transform(normalized_train_X)
            poly_model = LinearRegression()
            poly_model.fit(normalized_train_X, normalized_train_y)
            train_pred = poly_model.predict(normalized_train_X)

            normalized_test_X = poly.transform(normalized_test_X)
            test_pred = poly_model.predict(normalized_test_X)
            test_mse = mean_squared_error(normalized_test_y, test_pred)
            train_mse = mean_squared_error(normalized_train_y, train_pred)
            pred_model = test_df['model'].iloc[np.argmax(test_pred)]

            if pred_model == optimal[i]:
                print(benchmark)
                print(pred_model)
                # if length_optimal[i] > train_model_count:
                #     length_optimal[i] = train_model_count
                # found_optimal[i] = 0
                results.append([benchmark, train_models, test_mse, train_mse])

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=['Benchmark', 'Train Models', 'Test MSE', 'Train MSE'])
results_df.to_csv('benchmark_train_model_performance_all.csv', index=False)


    #         test_mse = mean_squared_error(normalized_test_y, test_pred)
    #         transformed_mse = 1/test_mse if test_mse > 0 else float('inf')
            
    #         average_normed_a.append(train_df['normed_a'].to_numpy())
    #         averaged_mse.append(transformed_mse)
    #     average_normed_a = np.array(average_normed_a)
    #     combined_normed_a.append(average_normed_a.mean(0))
    #     combined_mses.append(mean(averaged_mse))

    # # Remove duplicates and average the corresponding r2_scores
    # normed_a_mse_dict = [{} for _ in range(train_model_count)]
    # for normed_a_array, r2 in zip(combined_normed_a, combined_mses):
    #     for i, normed_a in enumerate(normed_a_array):
    #         if normed_a in normed_a_mse_dict[i]:
    #             np.append(normed_a_mse_dict[i][normed_a],r2)
    #         else:
    #             normed_a_mse_dict[i][normed_a] = np.array(r2)
                
    # for i in range(train_model_count):
    #     for k, v in normed_a_mse_dict[i].items():
    #         normed_a_mse_dict[i][k] = np.mean(v)
                
    # plt.figure(figsize=(10, 6))
    # plt.xlabel('normed_a')
    # plt.ylabel('test_r2')
    # plt.title('Test R2 vs normed_a')
    # plt.grid(True)
    # plt.legend()

    # for i in range(train_model_count):
    # # Sort the values for plotting
    #     sorted_indices = np.argsort(list(normed_a_mse_dict[i].keys()))
    #     unique_normed_a = np.array(list(normed_a_mse_dict[i].keys()))[sorted_indices]
    #     averaged_r2_scores = np.array(list(normed_a_mse_dict[i].values()))[sorted_indices]
    #     print(unique_normed_a)
    #     print(averaged_r2_scores)

    #     # Smooth the curve
    #     f_interp = interp1d(unique_normed_a, averaged_r2_scores, kind='cubic', fill_value="extrapolate")
    #     X_ = np.linspace(unique_normed_a.min(), unique_normed_a.max(), 2000)
    #     Y_ = f_interp(X_)
        
    #         # Set y-values to zero where there are no data points
    #     mask = np.ones_like(X_, dtype=bool)
    #     for x_val in unique_normed_a:
    #         mask[np.isclose(X_, x_val, atol=1e-3)] = False  # Adjust atol as needed for floating-point precision

    #     Y_[mask] = 0

    #     # Plot the results
    #     plt.plot(X_, Y_)
    # plt.show()
    # # exit()
import pandas as pd
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Select model combinations based on AC scores.")
    parser.add_argument('--data', type=str, choices=['A', 'C', 'AC', 'random', 'Ar'], required=True, help="Type of data to use for training.")
    parser.add_argument('--model', type=str, choices=['linear', 'polynomial'], required=True, help="Type of model to use for training.")
    parser.add_argument('--file_name', type=str, required=True, help="Name of the output CSV file.")

    args = parser.parse_args()

    df = pd.read_csv('/Users/shijiayang/Desktop/Vision_Feature_AC_private/visualizations/ablations_t.csv')

    # Define the parameters
    benchmarks = ["mmbench_en", "mme", "mmmu_val", "ok_vqa", "textvqa_val", "vizwiz_vqa_val", "scienceqa_img", "seed_image"]
    train_models = ["CLIP336", "CLIP224", "OpenCLIP", "DINOv2", "SDim", "SD1.5", "SDXL", "DiT", "SD3", "SD2.1", "SigLIP", "CLIP224+DINOv2", "CLIP336+DINOv2"]

    # Initialize a dictionary to store the results
    # results = {'Benchmarks': benchmarks}
    results = {'Models': train_models}

    column_names = df.columns.tolist()

    # Loop through the combinations and compute the average error rates
    # results['train_mse'] = []
    # results['train_r2'] = []

    for benchmark in benchmarks:
        data_perf_max = df[benchmark].max()
        data_perf_min = df[benchmark].min()
        data_a_max = df[f'{benchmark}_average'].max()
        data_a_min = df[f'{benchmark}_average'].min()
        data_c_max = df['corres'].max()
        data_c_min = df['corres'].min()

        df["normed_a"] = (df[f'{benchmark}_average'] - data_a_min) / (data_a_max - data_a_min)
        df["normed_c"] = (df['corres'] - data_c_min) / (data_c_max - data_c_min)

        train_df = df[df['model'].isin(train_models)]

        normalized_train_y = (train_df[benchmark] - data_perf_min) / (data_perf_max - data_perf_min)

        if args.data == 'AC':
            normalized_train_X = train_df[['normed_a', 'normed_c']]
        elif args.data == 'random' and args.model == 'polynomial':
            normalized_train_X = train_df[['normed_a', 'normed_c']]
            column_names = normalized_train_X.columns
            num_rows = len(normalized_train_X)
            normalized_train_X = pd.DataFrame(np.random.rand(num_rows, len(column_names)), columns=column_names)
        elif args.data == 'random' and args.model == 'linear':
            normalized_train_X = train_df[['normed_a']]
            column_names = normalized_train_X.columns
            num_rows = len(normalized_train_X)
            normalized_train_X = pd.DataFrame(np.random.rand(num_rows, len(column_names)), columns=column_names)
        elif args.data == 'A' and args.model == 'polynomial':
            normalized_train_X = train_df[['normed_a', 'normed_a']]
        elif args.data == 'A' and args.model == 'linear':
            normalized_train_X = train_df[['normed_a']]
        elif args.data == 'C' and args.model == 'polynomial':
            normalized_train_X = train_df[['normed_c', 'normed_c']]
        elif args.data == 'C' and args.model == 'linear':
            normalized_train_X = train_df[['normed_c']]
        elif args.data == 'Ar' and args.model == 'polynomial':
            normalized_train_X = train_df[['normed_a']].copy()  # Start with 'normed_a'
            # Generate a random column with the same number of rows
            random_column = pd.DataFrame(np.random.rand(len(normalized_train_X), 1), columns=['random'])
            # Concatenate the 'normed_a' and random column
            normalized_train_X = pd.concat([normalized_train_X, random_column], axis=1)

        # Polynomial regression (degree 2)
        if args.model == 'polynomial':
            poly = PolynomialFeatures(degree=2)
            print(normalized_train_X.shape)
            normalized_train_X = poly.fit_transform(normalized_train_X)
            print(normalized_train_X.shape)
        model = LinearRegression()
        model.fit(normalized_train_X, normalized_train_y)
        train_pred = model.predict(normalized_train_X)
        train_mse = mean_squared_error(normalized_train_y, train_pred)
        train_r2 = r2_score(normalized_train_y, train_pred)

        # results['train_mse'].append(train_mse)
        # results['train_r2'].append(train_r2)
        print(benchmark, train_r2)
        results[f'{benchmark}_A'] = df["normed_a"].copy()
        results[f'{benchmark}_C'] = df["normed_c"].copy()
        

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'/Users/shijiayang/Desktop/Vision_Feature_AC_private/visualizations/{args.file_name}.csv', index=False)

if __name__ == "__main__":
    main()

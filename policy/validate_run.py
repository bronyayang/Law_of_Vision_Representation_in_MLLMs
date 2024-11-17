import pandas as pd
import itertools
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


# Loop through the combinations and compute the average error rates
def validate_run(benchmark, train_models, top=1):
    df = pd.read_csv('/Users/shijiayang/Desktop/Vision_Feature_AC_private/visualizations/ablations_t.csv')
    all_models = ["CLIP336", "CLIP224", "OpenCLIP", "DINOv2", "SDim", "SD1.5", "SDXL", "DiT", "SD3", "SD2.1", "SigLIP", "CLIP224+DINOv2", "CLIP336+DINOv2"]

    optimal = {"mmbench_en": "CLIP224+DINOv2", "mme": "CLIP336", "mmmu_val": "OpenCLIP", "ok_vqa": "CLIP336+DINOv2", "textvqa_val": "CLIP336+DINOv2", "vizwiz_vqa_val": "CLIP336", "scienceqa_img": "CLIP336", "seed_image": "CLIP336+DINOv2"}
    
    train_models_set = set(train_models)
    # test_models = [model for model in all_models if model not in train_models_set]
    test_models = all_models


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
    pred_model = test_df['model'].iloc[np.argsort(test_pred)[-top:]]

    if optimal[benchmark] in pred_model.to_list():
        return True, pred_model
    else:
        return False, pred_model

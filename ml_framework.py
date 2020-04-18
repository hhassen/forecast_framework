#  This file contains the required functions to execute forecast_energy.py

import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer
from skits.feature_extraction import (AutoregressiveTransformer, SeasonalTransformer)

# Function definition
# =============


def df_load(file_name):
    #  Loading file_name and parsing the delivery_start column and return parsed dataframe
    df = pd.read_csv(file_name, sep="\t")
    timeDF = (pd.to_datetime(df['delivery_start'].str.strip(), format='%Y-%m-%d %H:%M:%S+00:00'))
    df['delivery_start'] = timeDF
    df['date'] = range(1, len(df) + 1)
    print("******\n LOADING ", file_name)
    print(" Data loaded from : ", df['delivery_start'].min(), " to : ", df['delivery_start'].max(), "\n")
    return df


def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def train_pipeline(pipeline, df, feature, n, m):
    # Train the pipeline to predict feature using the first n points and return trained pipeline

    N = n + m  # number of total points
    y = df[feature][:N].values
    X = y[:, np.newaxis]
    # train only on n first values
    print("******\n TRAINING THE MODEL \n******")
    print("Training model using data from : ", df['delivery_start'].min(), " to : ", df['delivery_start'][n])
    pipeline.fit(X[:n], y[:n])
    # forecast on the next m values
    y_pred = pipeline.forecast(y[:, np.newaxis], start_idx=n)
    RMSE_train_forecast = mean_squared_error(y[n:], y_pred[n:])
    print("RMSE of forecast : ", RMSE_train_forecast)
    export_fig(y, pipeline, n, 'train.png')
    return pipeline, RMSE_train_forecast


def pipeline_forecast(pipeline, df, feature, n, m):
    #  Forecast feature on the given df

    N = n + m  # number of total points
    y = df[feature][:N].values
    # forecast starting from index n to the next m values
    print("******\n FORECASTING ON TEST SET \n******")
    print("Forecasting from : ", df['delivery_start'][n], " to : ", df['delivery_start'][N])
    y_pred = pipeline.forecast(y[:, np.newaxis], start_idx=n)
    RMSE_test_forecast = mean_squared_error(y[n:], y_pred[n:])
    print("RMSE of forecast : ", RMSE_test_forecast)
    export_fig(y, pipeline, n, 'test.png')
    return RMSE_test_forecast


def export_fig(y, pipeline, n, filename):
    #  Export figure of y_true and y_prediction

    start_idx = n
    plt.plot(y, lw=2)
    plt.plot(pipeline.forecast(y[:, np.newaxis], start_idx=start_idx), lw=2)
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.plot((start_idx, start_idx), ylim, lw=4)
    plt.ylim(ylim)
    plt.legend(['y_true', 'y_pred', 'forecast start'])
    plt.savefig(filename)
    plt.close()
    print("Figure exported : ", filename)
    return None


def create_pipeline(model_config):
    # Create pipeline using config file parameters

    pipeline_steps = [
        ('pre_scaling', StandardScaler()),
        ('features', FeatureUnion([
            ('ar_transformer', AutoregressiveTransformer(num_lags=model_config['AUTOREGRESSIVE_NUM_LAGS'])),
            ('seasonal_transformer', SeasonalTransformer(seasonal_period=model_config['SAMPLES_PER_DAY'])
             )])),
        ('post_features_imputer', ReversibleImputer()),
    ]

    if model_config['LINEAR_MODEL_ENABLE']:
        pipeline_steps.append(('regressor', LinearRegression(fit_intercept=model_config['LINEAR_FIT_INTERCEPT'])))
        print("Using LinearRegresion regressor. Hyperparamaters :\n  fit_intercept = ", model_config['LINEAR_FIT_INTERCEPT'])
    elif model_config['ELASTICNET_MODEL_ENABLE']:
        pipeline_steps.append(('regressor', ElasticNet(fit_intercept=model_config['ELASTICNET_FIT_INTERCEPT'],
                                                       alpha=model_config['ELASTICNET_ALPHA'],
                                                       l1_ratio=model_config['ELASTICNET_L1RATIO'])))
        print("Using ElasticNet regressor. Hyperparamaters :\n  fit_intercept = ", model_config['ELASTICNET_FIT_INTERCEPT'],
              "\n  alpha = ", model_config['ELASTICNET_ALPHA'])
    elif model_config['RANDOMFOREST_MODEL_ENABLE']:
        pipeline_steps.append(('regressor', RandomForestRegressor(n_estimators=model_config['RANDOMFOREST_N_ESTIMATORS'],
                                                                  max_depth=model_config['RANDOMFOREST_MAX_DEPTH'])))
        print("Using RandomForest regressor. Hyperparamaters :\n  n_estimators = ", model_config['RANDOMFOREST_N_ESTIMATORS'],
              "\n  max_depth = ", model_config['RANDOMFOREST_MAX_DEPTH'])
    else:
        raise SyntaxError('Please enable one regression model in the config file')

    pipeline = ForecasterPipeline(pipeline_steps)
    return pipeline


def export_csv(model_config, output_data, csv_file):
    # export config data and forecasting errors to csv_file

    csv_columns = ['n', 'm', 'TARGET_FEATURE', 'SAMPLES_PER_DAY', 'AUTOREGRESSIVE_NUM_LAGS',
                   'LINEAR_MODEL_ENABLE', 'LINEAR_FIT_INTERCEPT', 'ELASTICNET_MODEL_ENABLE', 'ELASTICNET_FIT_INTERCEPT',
                   'ELASTICNET_ALPHA', 'ELASTICNET_L1RATIO', 'RANDOMFOREST_MODEL_ENABLE', 'RANDOMFOREST_N_ESTIMATORS',
                   'RANDOMFOREST_MAX_DEPTH', 'RMSE_train_forecast', 'RMSE_test_forecast']
    output = [
        {'n': output_data['n'], 'm': output_data['m'],
         'TARGET_FEATURE': output_data['TARGET_FEATURE'],
         'SAMPLES_PER_DAY': model_config['SAMPLES_PER_DAY'],
         'AUTOREGRESSIVE_NUM_LAGS': model_config['AUTOREGRESSIVE_NUM_LAGS'],
         'LINEAR_MODEL_ENABLE': model_config['LINEAR_MODEL_ENABLE'],
         'LINEAR_FIT_INTERCEPT': model_config['LINEAR_FIT_INTERCEPT'],
         'ELASTICNET_MODEL_ENABLE': model_config['ELASTICNET_MODEL_ENABLE'],
         'ELASTICNET_FIT_INTERCEPT': model_config['ELASTICNET_FIT_INTERCEPT'],
         'ELASTICNET_ALPHA': model_config['ELASTICNET_ALPHA'],
         'ELASTICNET_L1RATIO': model_config['ELASTICNET_L1RATIO'],
         'RANDOMFOREST_MODEL_ENABLE': model_config['RANDOMFOREST_MODEL_ENABLE'],
         'RANDOMFOREST_N_ESTIMATORS': model_config['RANDOMFOREST_N_ESTIMATORS'],
         'RANDOMFOREST_MAX_DEPTH': model_config['RANDOMFOREST_MAX_DEPTH'],
         'RMSE_train_forecast': output_data['RMSE_train_forecast'],
         'RMSE_test_forecast': output_data['RMSE_test_forecast']}
    ]
    try:
        with open(csv_file, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            if is_file_empty(csv_file):
                # the header of csv file must be written only once
                writer.writeheader()
            for data in output:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    return None


import json
from ml_framework import *

# Reading configuration
# =============
with open('config.json', 'r') as file:
    JSON_CONFIG = json.loads(file.read())
print("Config file loaded: OK!")

n = JSON_CONFIG['global']['n']
m = JSON_CONFIG['global']['m']
TARGET_FEATURE = JSON_CONFIG['global']['TARGET_FEATURE']
TRAIN_FILE = JSON_CONFIG['global']['TRAIN_FILE']
TEST_FILE = JSON_CONFIG['global']['TEST_FILE']

model_config = {
    'SAMPLES_PER_DAY': JSON_CONFIG['model']['SAMPLES_PER_DAY'],
    'AUTOREGRESSIVE_NUM_LAGS': JSON_CONFIG['model']['AUTOREGRESSIVE_NUM_LAGS'],
    'LINEAR_MODEL_ENABLE': JSON_CONFIG['model']['linear']['enable'],
    'LINEAR_FIT_INTERCEPT': JSON_CONFIG['model']['linear']['FIT_INTERCEPT'],
    'ELASTICNET_MODEL_ENABLE': JSON_CONFIG['model']['elasticnet']['enable'],
    'ELASTICNET_FIT_INTERCEPT': JSON_CONFIG['model']['elasticnet']['FIT_INTERCEPT'],
    'ELASTICNET_ALPHA': JSON_CONFIG['model']['elasticnet']['ALPHA'],
    'ELASTICNET_L1RATIO': JSON_CONFIG['model']['elasticnet']['L1RATIO'],
    'RANDOMFOREST_MODEL_ENABLE': JSON_CONFIG['model']['randomforest']['enable'],
    'RANDOMFOREST_N_ESTIMATORS': JSON_CONFIG['model']['randomforest']['N_ESTIMATORS'],
    'RANDOMFOREST_MAX_DEPTH': JSON_CONFIG['model']['randomforest']['MAX_DEPTH'],
}

# =============
#  Main function
# =============
#  Training the model on training dataset
df_train = df_load(TRAIN_FILE)
pipeline = create_pipeline(model_config)
pipeline, RMSE_train_forecast = train_pipeline(pipeline, df_train, TARGET_FEATURE, n, m)

# Loading test dataset and forecasting
df_test = df_load(TEST_FILE)
RMSE_test_forecast = pipeline_forecast(pipeline, df_test, TARGET_FEATURE, n, m)

# CSV output
csv_file = "forecast_performance.csv"
output_data = {'n': n, 'm': m, 'TARGET_FEATURE': TARGET_FEATURE, 'RMSE_train_forecast': RMSE_train_forecast,
               'RMSE_test_forecast': RMSE_test_forecast}
export_csv(model_config, output_data, csv_file)

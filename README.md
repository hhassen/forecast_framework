# forecast_framework
Forecast framework to make training ML models and forecasting on time series easier.


## Abstract
Automate training, testing and deployment of machine learning models to forecast time-series variable. This is done very easily through changing parameters in a configuration file. Therefore, no coding experience is required and user can experience with several ML models in a fast way while keeping track of the performance of every model used in a CSV file.

![Screenshot](https://i.imgur.com/9MXEmVu.png)
![Output CSV Screenshot](https://i.imgur.com/lKK2esN.png)

## How to run ?
Change `config.json` to include your train, test dataset and your target feature.
You may also need to rename the time feature to `delivery_start` or you can change `df_load` function in `ml_framework.py`

execute `forecast_energy.py` using python 3:
`python forecast_energy.py`

## How it works
 1. The user select a ML model and customize its parameters in the configuration file
 2. The script trains the model on the n time steps of the dataset and make a forecast on the next m time steps of the training dataset (to get prediction score on training dataset)
 3. The trained model now makes a forecast on the test dataset, using n time steps provided of the test dataset.
 4. The performance of this model, along with the configuration used are exported to a csv table.

## ML workflow
Our model is made of several steps:
 1. Pre-scaling step to get all the attributes to have the same scale
 2. Auto-regressive transformer to introduce the past data values as different input features
 3. Seasonal transformer to take into consideration the seasonality of the data. For short-term forecasts, a day can be considered as a season and for long-term forecasts, we can take a month or a year as a season.
 4. Reversible Imputer to make sure we do not have any NaN value in the features.
 5. A regressor, which is a ML model, that allows to make regression on the given data. As a start, I added a Linear regressor, ElasticNet regressor and RandomForest regressor.

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import logger  # Module for handling logs

log = logger.get_logger(__name__)  # Logger initialization


def imputation_per_category(df, columns_for_grouping, columns_for_imputation,
                            stat='median'):
    """Fills NULL values by applying an aggregation statistic per group

    :param df: (pandas.core.frame.Dataframe) Dataframe with some NULL values to be filled up
    :param columns_for_grouping: (list) Names of the columns used for aggregation
    :param columns_for_imputation: (list) Names of the columns with NULL values to be filled up
    :param stat: (str) Aggregation statistic. Choose between 'median', 'mean', 'max', 'min'

    :return: (pandas.core.frame.Dataframe) Dataframe after imputation
    """
    log.debug(f'Data imputation for {columns_for_imputation} based on {columns_for_grouping} \n Aggregating by {stat}')
    df[columns_for_imputation] = df[columns_for_imputation].\
        fillna(df.groupby(columns_for_grouping)[columns_for_imputation].transform(stat))
    log.debug('Data imputation complete')
    return df


def extract_temporal_features(df, timestamp_column='timestamp'):
    """ Explode a timestamp column into multiple time feature columns

    :param df: (pandas.core.frame.Dataframe) Dataframe with a timestamp column
    :param timestamp_column: (str) Name of the timestamp column

    :return: (pandas.core.frame.Dataframe) Dataframe with additional columns date, time, hour, month, dayofyear,
    dayofweek
    """
    df['date'] = df[timestamp_column].dt.date
    df['time'] = df[timestamp_column].dt.time
    df['hour'] = df[timestamp_column].dt.hour
    df['month'] = df[timestamp_column].dt.month
    df['dayofyear'] = df[timestamp_column].dt.dayofyear
    df['dayofweek'] = df[timestamp_column].dt.dayofweek
    return df


def dummifier(df, column, drop=True):
    """Performs one-hot encoding for a column in a dataframe

    :param df: (pandas.core.frame.Dataframe) Dataframe with column to be encoded
    :param column: (str) Name of the column to be encoded
    :param drop: (bool) True to drop the column after encoding. False to keep the column

    :return: (pandas.core.frame.Dataframe) Dataframe with column encoded
    """
    log.debug(f'One hot encoding for {column}')
    dummies = pd.get_dummies(df[column], prefix=column).astype('int64')
    df = pd.concat([df, dummies], axis=1)
    if drop:
        df.drop(columns=[column], inplace=True)
    return df


def resampler(df, column, undersampling=True):
    """Fix class imbalance in a dataframe

    :param df: (pandas.core.frame.Dataframe) Dataframe to be balanced
    :param column: (str) Name of the column with the facets (classes) to balance
    :param undersampling: (bool) True if the resampling method is undersampling. False if it is oversampling

    :return: (pandas.core.frame.Dataframe) Dataframe with balanced facets
    """
    log.debug(f'Starting resampling based on {column}...')
    if undersampling:
        resampler_ = RandomUnderSampler(sampling_strategy='all', random_state=42)
    else:
        resampler_ = RandomOverSampler(sampling_strategy='all', random_state=42)
    X_resampled, y_resampled = resampler_.fit_resample(
        df.drop(columns=[column]),
        df[column])
    log.debug('Resampling complete')
    return pd.concat([X_resampled, y_resampled], axis=1)


def prepare_building_metadata(building_metadata_path):
    """Cleans and prepares building_metadata.csv

    :param building_metadata_path: (str) Path of the file building_metadata.csv

    :return: (pandas.core.frame.Dataframe) Dataframe after preprocessing
    """
    log.debug('Preparing building_metadata')
    building_metadata_df = pd.read_csv(building_metadata_path)
    # Check for columns with NULL values
    log.debug(f'Initial percentage of NULL values: \n{building_metadata_df.isna().sum()/len(building_metadata_df)}')
    # Drop feature with more than 75% NULL values
    building_metadata_df.drop(columns=['floor_count'], inplace=True)
    # Data imputation for year_built. First, we use the median year_built of the buildings with the same site_id
    # and primary_use
    columns_for_imputation = ['year_built']
    columns_for_grouping = ['site_id', 'primary_use']
    building_metadata_df = imputation_per_category(building_metadata_df,
                                                   columns_for_grouping,
                                                   columns_for_imputation,
                                                   stat='median')
    log.debug(f'Updated percentage of NULL values: \n{building_metadata_df.isna().sum()/len(building_metadata_df)}')
    # Data imputation for year_built for those buildings failing the first imputation round
    building_metadata_df['year_built'].fillna(
        building_metadata_df['year_built'].median(), inplace=True)
    log.debug(f'Final percentage of NULL values: \n{building_metadata_df.isna().sum()/len(building_metadata_df)}')
    log.debug('building_metadata preparation complete')
    return building_metadata_df


def prepare_weather_data(weather_data_path):
    """Cleans and prepares weather_data.csv

    :param weather_data_path: (str) Path of the file weather_data.csv

    :return: (pandas.core.frame.Dataframe) Dataframe after preprocessing
    """
    log.debug('Preparing weather_data...')
    weather_data_df = pd.read_csv(weather_data_path, parse_dates=['timestamp'])
    log.debug(f'Initial percentage of NULL values: \n{weather_data_df.isna().sum()/len(weather_data_df)}')
    weather_data_df = extract_temporal_features(weather_data_df,
                                                timestamp_column='timestamp')
    # Data imputation for weather variables. First, we use the mean value of the instances with the same site_id
    # and date
    columns_for_imputation = ['air_temperature', 'cloud_coverage',
                              'dew_temperature', 'precip_depth_1_hr',
                              'sea_level_pressure', 'wind_direction', 'wind_speed']
    columns_for_grouping = ['site_id', 'date']
    weather_data_df = imputation_per_category(weather_data_df,
                                              columns_for_grouping,
                                              columns_for_imputation,
                                              stat='mean')
    log.debug(f'Updated percentage of NULL values: \n{weather_data_df.isna().sum()/len(weather_data_df)}')
    # Further data imputation for weather variables. Now, we assign the mean value per site_id and month
    columns_for_grouping = ['site_id', 'month']
    weather_data_df = imputation_per_category(weather_data_df,
                                              columns_for_grouping,
                                              columns_for_imputation,
                                              stat='mean')
    log.debug(f'Updated percentage of NULL values: \n{weather_data_df.isna().sum()/len(weather_data_df)}')
    # Final imputation assigning the median value per column
    columns_with_nan = ['cloud_coverage', 'precip_depth_1_hr', 'sea_level_pressure']
    weather_data_df[columns_with_nan] = weather_data_df[columns_with_nan]. \
        fillna(weather_data_df[columns_with_nan].median())
    log.debug(f'Final percentage of NULL values: \n{weather_data_df.isna().sum()/len(weather_data_df)}')
    log.debug('weather_data preparation complete')
    return weather_data_df


def prepare_building_meter_readings(building_meter_readings_path):
    """Cleans and prepares building_meter_readings.csv

    :param building_meter_readings_path: (str) Path of the file building_meter_readings.csv

    :return: (pandas.core.frame.Dataframe) Dataframe after preprocessing
    """
    log.debug('Preparing building_meter_readings...')
    building_meter_readings_df = pd.read_csv(building_meter_readings_path, parse_dates=['timestamp'])
    # Some buildings use simultaneously more than one energy type. We want to add that information to each instance
    log.debug('Flagging co-occurrences of multiple meters')
    energy_type_usage_indicator = dummifier(
        building_meter_readings_df[['building_id', 'timestamp', 'meter']], 'meter')
    energy_type_usage_per_timestamp = energy_type_usage_indicator.groupby(
        ['building_id', 'timestamp']).sum().reset_index()
    energy_type_usage_per_timestamp.columns = ['building_id', 'timestamp', 'has_meter_0', 'has_meter_1', 'has_meter_2',
                                               'has_meter_3']
    building_meter_readings_df = pd.merge(
        left=building_meter_readings_df,
        right=energy_type_usage_per_timestamp,
        on=['building_id', 'timestamp'],
        how='inner'
        )
    log.debug('building_meter_readings preparation complete')
    return building_meter_readings_df

    
def data_preparation(building_metadata_path='./data/building_metadata.csv',
                     weather_data_path='./data/weather_data.csv',
                     building_meter_readings_path='./data/building_meter_readings.csv',
                     training=True, log_trans=False, undersampling=True):
    """
    Creates dataframe ready for modeling

    :param building_metadata_path: (str) Path of the file building_metadata.csv
    :param weather_data_path: (str) Path of the file weather_data.csv
    :param building_meter_readings_path: (str) Path of the file building_meter_readings.csv
    :param training: (bool) True if preparing the dataframe for model training. False otherwise
    :param log_trans: (bool) True if we want to apply log transformation to the target variable. False otherwise
    :param undersampling: (bool) True if the resampling method is undersampling. False if it is oversampling

    :return: (pandas.core.frame.Dataframe) Dataframe with only numerical features, no NULL values, balanced by meter
    """
    log.info('Starting data preparation...')
    building_metadata_df = prepare_building_metadata(building_metadata_path)
    weather_data_df = prepare_weather_data(weather_data_path)
    building_meter_readings_df = prepare_building_meter_readings(building_meter_readings_path)
    # Merge building_meter_readings with building_metadata
    df = pd.merge(left=building_meter_readings_df,
                  right=building_metadata_df,
                  on='building_id',
                  how='left')
    # Fix incorrect unit from original dataset (https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261)
    df.loc[(df['site_id'] == 0) & (df['meter'] == 0), 'meter'] = df['meter']*0.2931
    # Merge with weather_data
    df = pd.merge(left=df,
                  right=weather_data_df,
                  on=['site_id', 'timestamp'],
                  how='left')
    # Remove redundant/unnecessary columns
    unnecessary_columns = ['timestamp', 'time', 'building_id', 'date']
    df.drop(columns=unnecessary_columns, inplace=True)
    if training:
        # Remove outliers
        df = df[abs(df['meter_reading']-df['meter_reading'].mean()) < df['meter_reading'].std()]
        if log_trans:
            df['meter_reading'] = np.log1p(df['meter_reading'])
        # Check for imbalance in meter category
        log.debug(f'Class imbalance for meter:\n{df.meter.value_counts()}')
        # Undersample 
        df = resampler(df, 'meter', undersampling)
    # One hot encoding for site_id and primary_use
    object_columns = ['site_id', 'primary_use', 'meter']
    for column in object_columns:
        df = dummifier(df, column)
    df.rename(columns={'meter_0.0': 'meter_0_ind', 
                       'meter_1.0': 'meter_1_ind',
                       'meter_2.0': 'meter_2_ind',
                       'meter_3.0': 'meter_3_ind'},
              inplace=True)
    log.info('Data preparation complete')
    return df


def evaluation(y_test_, y_pred_):
    """
    Calculates and logs evaluation metrics: MSE, RMSE, MAE, R^2, Normalized RMSE

    :param y_test_: (pandas.core.series.Series) Series with the actual values of the target variable
    :param y_pred_: (pandas.core.series.Series) Series with the predicted values of the target variable

    :return: None
    """
    log.info('Model evaluation: ')
    mse = mean_squared_error(y_test_, y_pred_)
    mae = mean_absolute_error(y_test_, y_pred_)
    r2 = r2_score(y_test_, y_pred_)
    nrmse = 100 * np.sqrt(mse) / (max(y_test_) - min(y_test_))
    log.info("MSE: %.2f" % mse)
    log.info("RMSE: %.2f" % np.sqrt(mse))
    log.info("MAE: %.2f" % mae)
    log.info("R^2: %.2f" % r2)
    log.info("Normalized RMSE: %.2f%%" % nrmse)
    return

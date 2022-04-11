import platform; print(platform.platform())
import sys; print("Python", sys.version)
import os
import pandas as pd
import numpy as np
from uuid import uuid4
from joblib import load

import logger  # Module for handling logs
log = logger.setup_applevel_logger(
    file_name='inference.log',
    is_debug=True)  # Our logger must be initialized prior to loading additional modules
import lib  # All our helper functions are here


def inference():
    # Load directory paths for persisting data
    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE = os.environ["MODEL_FILE"]
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
    DATA_DIR = os.environ['DATA_DIR']
    WEATHER_DATA_TEST_FILE = os.environ["WEATHER_DATA_TEST_FILE"]
    WEATHER_DATA_TEST_PATH = os.path.join(DATA_DIR, WEATHER_DATA_TEST_FILE)
    BUILDING_METER_READINGS_TEST_FILE = os.environ["BUILDING_METER_READINGS_TEST_FILE"]
    BUILDING_METER_READINGS_TEST_PATH = os.path.join(DATA_DIR, BUILDING_METER_READINGS_TEST_FILE)
    DEFAULTS_FILE = 'defaults.csv'
    DEFAULTS_PATH = os.path.join(DATA_DIR, DEFAULTS_FILE)
    PREDICTIONS_FILE = 'prediction.csv'
    PREDICTIONS_PATH = os.path.join(DATA_DIR, PREDICTIONS_FILE)
    
    RUN_SERIAL = str(uuid4())
    log_trans = False
    
    log.info('Starting inference...')
    # Import and prepare data
    df = lib.data_preparation(weather_data_path=WEATHER_DATA_TEST_PATH,
                              building_meter_readings_path=BUILDING_METER_READINGS_TEST_PATH,
                              training=False)
    # Fill missing columns/values with default values
    defaults_df = pd.read_csv(DEFAULTS_PATH)
    df_filled = defaults_df.loc[defaults_df.index.repeat(len(df))].reset_index(drop=True)
    df_filled.update(df)
    # Import model
    model1 = load(MODEL_PATH)
    # Predict
    y_pred = model1.predict(data=df_filled)
    if log_trans:
        y_pred = np.expm1(y_pred)
    y_pred = np.maximum(0, y_pred)
    # Export predictions
    building_meter_readings_test_df = pd.read_csv(BUILDING_METER_READINGS_TEST_PATH)
    building_meter_readings_test_df['predicted'] = y_pred
    building_meter_readings_test_df.to_csv(PREDICTIONS_PATH, index=False)
    log.info(f'Successfully saved predictions at {PREDICTIONS_PATH}')
    log.info(f'Inference complete. Run serial: {RUN_SERIAL}')
    
    
if __name__ == '__main__':
    inference()

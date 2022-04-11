import platform; print(platform.platform())
import sys; print("Python", sys.version)

import os
import numpy as np
import pandas as pd
from uuid import uuid4
from joblib import dump
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

import logger  # Module for handling logs
log = logger.setup_applevel_logger(
    file_name='train.log',
    is_debug=True)  # Our logger must be initialized prior to loading additional modules
import lib  # All our helper functions are here


def train():

    # Load directory paths for persisting model and data
    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE = os.environ["MODEL_FILE"]
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
    
    DATA_DIR = os.environ['DATA_DIR']
    DEFAULTS_FILE = 'defaults.csv'
    DEFAULTS_PATH = os.path.join(DATA_DIR, DEFAULTS_FILE)
        
    RUN_SERIAL = str(uuid4())
    log_trans = False
    undersampling = True

    log.info('Starting training model...')
    # Import and prepare data
    df = lib.data_preparation(log_trans=log_trans, undersampling=undersampling)
    # Create training and test sets
    X = df.drop(columns=['meter_reading'])
    y = df['meter_reading']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42)
    train_pool = Pool(X_train, y_train)
    # Define model
    log.info('Starting ensemble model...')
    model1 = CatBoostRegressor(loss_function='RMSE', early_stopping_rounds=50,
                               verbose=250)
    # Set up grid for hyperparameter tuning and cross-validation
    grid = {'iterations': [1000, 5000],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.5],
            'l2_leaf_reg': [2, 20, 50]}
    model1.grid_search(grid, train_pool, cv=3)
    log.info('Ensemble model complete')
    log.info(f'Model parameters: \n{model1.get_params()}')
    # Evaluate model
    y_pred = model1.predict(data=X_test)
    if log_trans:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
    y_pred = np.maximum(0, y_pred)
    lib.evaluation(y_test, y_pred)
    # Feature importance
    log.info(f'Feature importance: \n{model1.get_feature_importance(prettified=True)}')
    # Save model
    dump(model1, MODEL_PATH)
    log.info(f'Successfully saved model at {MODEL_PATH}')
    # Save default values for future predictions with incomplete data
    pd.DataFrame(X.median()).transpose().to_csv(DEFAULTS_PATH, index=False)
    log.info(f'Training model complete. Run serial: {RUN_SERIAL}')

        
if __name__ == '__main__':
    train()

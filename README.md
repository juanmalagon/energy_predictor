## About the project

Solution for ASHRAE - Great Energy Predictor III
<br />
<p align="center">
<img src=https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python alt="Python"/>
<img src=https://img.shields.io/badge/anaconda-square?logo=anaconda&label=Made%20With&style=for-the-badge alt="Anaconda"/>
<img src=https://img.shields.io/badge/Made%20with-Docker-blue?style=for-the-badge&logo=dockeralt="Docker"/>
</p>

## Running the implementation
1. Copy the file `train.csv` (647.2 MB) to the folder `./data` (this file is too large to be shared via GitHub). Name the file as `building_meter_readings.csv`.
2. Create an image by running: `docker build -t ashrae -f Dockerfile .`
3. Run a container for training by running: `docker run -it --rm --name runtrain ashrae`
4. Run a container for inference by running: `docker run -it --rm --name runinference ashrae python3 inference.py`

## Training
The script `train.py` reads the files:
* `.data/building_meter_readings.csv`
* `.data/building_metadata.csv`
* `.data/weather_data.csv`

Then, it creates the CatBoost model and stores it as `reg_cat.joblib`. Additionally, this script creates a file `.data/defaults.csv` with default values which are used later for predictions (whenever there are missing values).

Logs for the run, including evaluation results, are stored in `train.log`.

## Inference
Following the convention of https://www.kaggle.com/c/ashrae-energy-prediction, the `inference.py` script reads two files:
* `.data/building_meter_readings_test.csv`
* `.data/weather_data_test.csv`

Then, it calculates the prediction and stores them as `.data/prediction.csv`. 

Logs for the run are stored in `inference.log`.

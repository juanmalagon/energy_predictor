FROM python:3.9.10

WORKDIR /app

ENV MODEL_DIR=/app
ENV MODEL_FILE=reg_cat.joblib

ENV DATA_DIR=/app/data
ENV WEATHER_DATA_TEST_FILE=weather_data_test.csv
ENV BUILDING_METER_READINGS_TEST_FILE=building_meter_readings_test.csv

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./train.py" ]

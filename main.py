"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import argparse
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from data.data import process_data, get_coords
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def initialise():
    global lag
    global data
    global lstm
    global gru
    global saes

    #Define some setting
    lag = 3
    data = 'data/Scats Data October 2006.csv'

    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')



def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

    

def main():
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    #Call data.py process_data function for testing data
    _, _, x_test, y_test, scaler = process_data(data, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
        else:
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(x_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print("---------------------------------name-----------------------------------")
        print(name)
        print("--------------------------------Predict---------------------------------")

        print(predicted)
        print("--------------------------------x test---------------------------------")
        print(x_test)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)



def convert_str_to_minutes(time):
    # Split the time string into hours and minutes
    hours, minutes = map(int, time.split(':'))

    # Convert hours to minutes and add the minutes
    total_minutes = int(hours) * 60 + int(minutes)

    return total_minutes

def predict_traffic_flow(latitude, longitude, time, date, model):
    # convert time string to minutes
    time = convert_str_to_minutes(time)

    # convert time, so its the same as df['Time'] in data.py. which is split in 15 min segments
    time = time / 1440
    time = int(time)

    # Transform latitude and longitude using respective scalers
    _, _, _, _, scaler = process_data(data, lag)
    scaled_lat = scaler.transform(np.array(latitude).reshape(1, -1))[0][0]
    scaled_long = scaler.transform(np.array(longitude).reshape(1, -1))[0][0]

    # Prepare test data
    x_test = np.array([[scaled_lat, scaled_long, time]])
    print(scaled_lat.dtype)
    print(scaled_long.dtype)
    #print(date.dtype) str
    #print(time.dtype) int

    # Reshape x_test based on the chosen model
    if model in ['saes']:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    else:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        

    # Map the string name of the model to the actual model object
    model_map = {
        'lstm': lstm,
        'gru': gru,
        'saes': saes,
    }

    # Select the desired model
    selected_model = model_map.get(model.lower())
    if selected_model is None:
        raise ValueError(f"Unsupported model: {model}")

    print(f"Select {model}") ####

    # Predict using the selected model
    predicted = selected_model.predict(x_test)

    # Transform the prediction using the flow_scaler to get the actual prediction
    final_prediction = scaler.inverse_transform(predicted)
    
    return final_prediction


if __name__ == '__main__':
    initialise()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scats",
        default="4034",
        help="SCATS site number.")
    parser.add_argument(
        "--direction",
        default="E",
        help="The approach to the site (N, S, E, W, NE, NW, SE)")
    parser.add_argument(
        "--time",
        default="13:30",
        help="The time in 24 hr notation")
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to use for prediction (lstm, gru, saes)")
    args = parser.parse_args()

    #main()

    lat, long = get_coords(data, args.scats, args.direction)
    print(lat, long)

    flow_prediction = predict_traffic_flow(latitude=lat, longitude=long, date=args.date, time=args.time, model=args.model)

    print(flow_prediction)


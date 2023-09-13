""" Traffic Flow Prediction with Neural Networks(SAEs, LSTM, GRU). """

# Import necessary libraries and modules
import math
import warnings
import os
import sys
import argparse

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

from data.data import process_data
from keras.models import load_model
from tensorflow.keras.utils import plot_model
warnings.filterwarnings("ignore")


# Define a function to calculate Mean Absolute Percentage Error (MAPE)
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

# Define a function to evaluate regression model performance
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


# Define a function to plot true and predicted traffic flow data
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


# Define the main function for evaluating and visualizing neural network models
def main():
    # Load pre-trained neural network models
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scats",
        default=970,
        help="SCATS site number.")
    parser.add_argument(
        "--junction",
        default=1,
        help="The approach to the site.")
    args = parser.parse_args()
    models = []
    names = ['LSTM', 'GRU', 'SAEs']

    count = 0
    for name in names:
        file = "model/{0}/{1}/{2}.h5".format(name.lower(), args.scats, args.junction)
        if os.path.exists(file):
            models.append(load_model(file))
        else:
            names.pop(count)
        count += 1



    lag = 12
    #file1 = 'data/Scats Data October 2006.csv'                                                  # this is the data file location 
    #file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))             # reshapes the SEAs mdoel to be 2D
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'                                                # stores a visualisation of the model architecture
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)


if __name__ == '__main__':
    main()

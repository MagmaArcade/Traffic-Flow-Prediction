"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import argparse
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from data.data import process_data
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from graphics import egi, KEY
from pyglet import window, clock
from pyglet.gl import *
from vector2d import Vector2D
from world import World
from path import Path
from math import sqrt
import argparse



def init_models():
    global lstm
    global gru
    global saes
    global flow_scaler
    global lat_scaler
    global long_scaler

    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    #_, _, flow_scaler, lat_scaler, long_scaler = process_data()

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

    

def on_key_press(symbol, modifiers):
    #Calculates up to 5 routes from Origin to Destination
    if (symbol== KEY.SPACE):
        #Resets line colouring
        world.reset()
        #Possible amount of additional routes from Origin to Destination
        extraPaths = 4
        paths = [Path([],world.origin,False,0)]

        #Optimal path completes its search
        paths[0].search(world.destination,world.scats,world.xM)
        #Optimal path data used to make the suboptimal path beginings
        paths = paths + paths[0].makePaths(extraPaths,world.destination, world.scats,world.xM)
        i = 0
        for path in paths:
            #If check so only suboptimal paths are searched
            if (i>0):
                path.search(world.destination,world.scats,world.xM)
                if (path.arrive == True):
                    world.successes.append(path)
            #If check so that if optimal path fails to get to Destination, its not auto added to the successes
            if (i == 0 and path.arrive == True):
                world.successes.append(path)
            i += 1
        #Order successful paths by time
        world.successes.sort(key=lambda x: x.distance, reverse=False)
        #Display new routes
        world.switchRoute()

    #Switch currently displayed route from Origin to Destination
    elif (symbol == KEY.Q):
        world.switchRoute()

    #Origin/Destination toggle
    elif (symbol == KEY.TAB):
        if (world.toggle):    
            world.toggle = False
        else:
            world.toggle = True

    
def on_mouse_press(x, y, button, modifiers):
    #Couldn't get right click to work, so changing Origin/Destination is decided by a toggle
    if (button == 1 and world.toggle == False):
        for scat in world.scats:
            if (sqrt((scat.pos.x-x)*(scat.pos.x-x)+(scat.pos.y-y)*(scat.pos.y-y)) < 10):
                world.resetOrigin()
                world.origin = scat.SCAT
                scat.color = "ORANGE"

    elif (button == 1 and world.toggle):
        for scat in world.scats:
            if (sqrt((scat.pos.x-x)*(scat.pos.x-x)+(scat.pos.y-y)*(scat.pos.y-y)) < 10):
                world.resetDestination()
                world.destination = scat.SCAT
                scat.color = "BLUE"


def on_resize(cx, cy):
    world.cx = cx
    world.cy = cy



def main():
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    #Define some setting
    lag = 12
    data = 'data/Scats Data October 2006.csv'

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
        print("---------------------------------name----------------------------------------------")
        print(name)
        print("--------------------------------Predict---------------------------------")

        print(predicted)
        print("--------------------------------x test---------------------------------")
        print(x_test)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scats",
        default=970,
        help="SCATS site number.")
    parser.add_argument(
        "--junction",
        default=1,
        help="The approach to the site.")
    parser.add_argument(
        "--time",
        default=970,
        help="The time")
    parser.add_argument(
        "--day",
        default=1,
        help="The day of the week")
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to use for prediction (lstm, gru, saes)")
    args = parser.parse_args()


    # create a pyglet window and set glOptions
    win = window.Window(width=850, height=650, vsync=True, resizable=True)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # needed so that egi knows where to draw
    egi.InitWithPyglet(win)
    # prep the fps display
    fps_display = window.FPSDisplay(win)
    # register key and mouse event handlers
    win.push_handlers(on_key_press)
    win.push_handlers(on_mouse_press)
    win.push_handlers(on_resize)

    # create a world for agents
    world = World(850, 650)
    

    while not win.has_exit:
        win.dispatch_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # show nice FPS bottom right (default)
        delta = clock.tick()
        #world.update(delta)
        world.render()
        #fps_display.draw()
        # swap the double buffer
        win.flip()

    main()

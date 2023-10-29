"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, x_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        x_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        x_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)


    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, x_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        x_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = x_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input, p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, x_train, y_train, name, config)


def main(argv):
    #Define the input arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="The model to train.")
    args = parser.parse_args()

    #Define some setting
    lag = 4
    config = {"batch": 256, "epochs": 2}
    data = 'data/Scats Data October 2006.csv'

    #Call data.py process_data function
    x_train, y_train, _, _, _ = process_data(data, lag)

    #
    if args.model == 'lstm':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        m = model.get_lstm([4, 64, 64, 1])
        train_model(m, x_train, y_train, args.model, config)
    if args.model == 'gru':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        m = model.get_gru([4, 64, 64, 1])
        train_model(m, x_train, y_train, args.model, config)
    if args.model == 'saes':
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        m = model.get_saes([4, 400, 400, 400, 1])
        train_seas(m, x_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)

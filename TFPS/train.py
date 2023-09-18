""" Train the NN model """

# Import necessary libraries and modules
import os
import sys
import warnings
import argparse

import numpy as np
import pandas as pd

# Import custom functions and classes
from data.data import process_data, read_data
from data.scats import ScatsDB
from model import model
from config import get_setting
from keras.models import Model

# Ignore warning messages
warnings.filterwarnings("ignore")


# Define a function to train a single neural network model
def train_model(model, x_train, y_train, name, scats, junction, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        x_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        scats: integer, then number of the SCATS site.
        junction: integer, the VicRoads internal number representing the location.
        config: Dict, parameter for train.
    """
    # Compile the model with loss and optimizer
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    # Train the model and store the training history
    hist = model.fit(
        x_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    # Save the trained model to a file and export the training history to a CSV file
    os.makedirs("model/{0}/{1}".format(name, scats))
    model.save("model/{0}/{1}/{2}.h5".format(name, scats, junction))
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv("model/{0}/{1}/{2} loss.csv".format(name, scats, junction), encoding='utf-8', index=False)



# Define a function to train Stacked Autoencoders (SAEs)
def train_seas(models, x_train, y_train, name, scats, junction, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        x_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        scats: integer, then number of the SCATS site.
        junction: integer, the VicRoads internal number representing the location.
        config: Dict, parameter for train.
    """

    temp = x_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input,
                                       p.get_layer('hidden').output)
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

    train_model(saes, x_train, y_train, name, scats, junction, config)


# Define the main function for training neural network models
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(        # define Model
        "--model",
        default="lstm",
        help="Model to train.")
    parser.add_argument(        # define SCAT Number
        "--scats",
        default="all",
        help="SCATS site number.")
    parser.add_argument(        # define the Junction
        "--junction",
        default="all",
        help="The approach to the site.")
    args = parser.parse_args()

    # Check if the database file exists
if os.path.exists("data/{0}".format(get_setting("database"))):
    # Create a connection to the SCATS database
    with ScatsDB() as s:
        # Get a list of all SCATS site numbers in the database
        scats_numbers = s.get_all_scats_numbers()
        
        # Check if the user specified a specific SCATS site, if yes, update the list
        if args.scats != "all":
            scats_numbers = [args.scats]
        
        # Iterate over each SCATS site
        for scats in scats_numbers:
            # Get a list of approaches (junctions) for the current SCATS site
            junctions = s.get_scats_approaches(scats)
            
            # Check if the user specified a specific approach, if yes, update the list
            if args.junction != "all":
                junctions = [args.junction]
            
            # Define lag (time lag) and configuration parameters for training
            lag = 12
            config = {"batch": 256, "epochs": 2}
            
            # Iterate over each approach (junction) for the current SCATS site
            for junction in junctions:
                # Process the training data for the current SCATS site and approach
                x_train, y_train, scaler = process_data(scats, junction, lag)
                
                # Check the selected model type and train the model accordingly
                if args.model == 'lstm':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    m = model.get_lstm([12, 64, 64, 1])
                    train_model(m, x_train, y_train, args.model, scats, junction, config)
                if args.model == 'gru':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    m = model.get_gru([12, 64, 64, 1])
                    train_model(m, x_train, y_train, args.model, scats, junction, config)
                if args.model == 'saes':
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
                    m = model.get_saes([12, 400, 400, 400, 1])
                    train_seas(m, x_train, y_train, args.model, scats, junction, config)
else:
    # If the database file doesn't exist, read the data from an Excel file
    read_data("data/Scats Data October 2006.xls")

    
if __name__ == '__main__':
    main(sys.argv)

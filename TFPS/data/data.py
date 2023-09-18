""" Processing the data """

# Import necessary libraries and modules
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
#from time import gmtime, strftime
from data.scats import ScatsDB


# Define a function to read and process VicRoads data from an Excel file
def read_data(data):
    # Read the Excel file into a DataFrame, skipping the first row and parsing dates
    dataset = pd.read_excel(data, sheet_name='Data', skiprows=1, parse_dates=['Date'], date_parser=format_date, nrows=200)
    df = pd.DataFrame(dataset)

    current_scats = None
    current_junction = None

    # Open a connection to the ScatsDB
    with ScatsDB() as db:
        for row in df.itertuples():
            if row[1] != current_scats:
                current_scats = row[1]
                current_junction = row[8]
                db.insert_new_scats(current_scats, current_junction, row[4], row[5])
            else:
                if row[8] != current_junction:
                    current_junction = row[8]
                    db.insert_new_scats(current_scats, current_junction, row[4], row[5])

            for i in range(96):
                current_time = row[10] + " " + format_time(i)
                value = row[11 + i]
                db.insert_scats_data(current_scats, current_junction, current_time, value)

# Define a function to process VicRoads data for training and testing
def process_data(scats_number, junction, lags):
    """Process data
    Reshape and split VicRoads data.

    # Arguments
        scats_number: integer, the SCATS site number.
        junction: integer, the VicRoads internal number representing the location.
        lags: integer, time lag.

    # Returns
        x_train: ndarray, training input data.
        y_train: ndarray, training target data.
        x_test: ndarray, testing input data.
        y_test: ndarray, testing target data.
        scaler: MinMaxScaler, a data scaling object.
    """
    # Open a connection to the ScatsDB
    with ScatsDB() as s:
        # Retrieve volume data for training
        volume_training = s.get_scats_volume(scats_number, junction)
        
        # Testing using the remaining days of the month
        volume_testing = volume_training[2016:]
        
        # Training using the first 3 weeks
        volume_training = volume_training[:2015]

        # Create a MinMaxScaler to scale the data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(volume_training.reshape(-1, 1))
        
        # Apply scaling to the training and testing data
        flow1 = scaler.transform(volume_training.reshape(-1, 1)).reshape(1, -1)[0]
        flow2 = scaler.transform(volume_testing.reshape(-1, 1)).reshape(1, -1)[0]

        train, test = [], []
        
        # Prepare the training data with lagged sequences
        for i in range(lags, len(flow1)):
            train.append(flow1[i - lags: i + 1])
        
        # Prepare the testing data with lagged sequences
        for i in range(lags, len(flow2)):
            test.append(flow2[i - lags: i + 1])

        train = np.array(train)
        test = np.array(test)
        
        # Shuffle the training data
        np.random.shuffle(train)

        # Split the training and testing data into input (X) and target (y) variables
        x_train = train[:, :-1]  # All columns except the last one
        y_train = train[:, -1]   # The last column
        x_test = test[:, :-1]    # All columns except the last one
        y_test = test[:, -1]     # The last column

        return x_train, y_train, x_test, y_test, scaler

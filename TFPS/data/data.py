""" Processing the data """

# Import necessary libraries and modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Define a function to process and prepare the data for training and testing
def process_data(data, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """

    # Read the CSV data file into a DataFrame and fill NaN values with zeros
    df1 = pd.read_csv(data, encoding='utf-8').fillna(0)

    # Create a copy of the DataFrame for processing test data
    df2 = df1.copy()

    # Split the data into train and test sets, keeping 2/3 for training and 1/3 for testing
    df2.drop(df2.index[-1], inplace=True)  # Remove the last row from the copy (test data)
    i = df1.shape[0] - 1
    while i > 0:
        i -= 1
        if (i % 3) == 0:
            df1.drop(df1.index[i], inplace=True)  # Remove every third row for training data
      

    # Extract the flow data (V00 to V95) and scale it using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1.loc[:,'V00':'V95'].values.reshape(-1, 1)) 
    flow1 = scaler.transform(df1.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []   # Initialize empty lists to store training and testing data

    # Prepare training data by creating lags (time intervals)
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])  # Create lagged sequences of data

    # Prepare testing data by creating lags (time intervals)
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])  # Create lagged sequences of data

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    # Split the training and testing data into input (X) and target (y) variables
    X_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column
    X_test = test[:, :-1]    # All columns except the last one
    y_test = test[:, -1]     # The last column


    return X_train, y_train, X_test, y_test, scaler

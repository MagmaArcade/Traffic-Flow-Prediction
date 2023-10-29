"""
Processing the data
"""
#import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression


def process_data2(data):
    # Read in CSV file
    df1 = pd.read_csv(data, encoding='utf-8').fillna(0) 

    i = df1.shape[0] -1
    pr = 0000
    #Shortens the DF to remove duplicates
    while (i > -1):
        cr = df1.loc[i, 'SCATS_Number']
        if (cr == pr):
            df1 = df1.drop([i])
        pr = cr
        i -=1
    
    df1 = df1.reset_index(drop=True)

    return df1

def get_coords(data, scats, junction):
    # Read data file
    df = pd.read_csv(data, encoding='utf-8').fillna(0)

    # Filter the DataFrame around the SCATS, making a new DataFrame that just has this SCAT
    filtered_df = df[df['SCATS Number'] == int(scats)]

    # Remove duplicates if there are any
    filtered_df = filtered_df.drop_duplicates(subset=['NB_LATITUDE', 'NB_LONGITUDE'])

    # Get the lat and long postition for the juction
    lat = filtered_df['NB_LATITUDE'].iloc[junction]
    long = filtered_df['NB_LONGITUDE'].iloc[junction]

    return lat, long

def process_data(data, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        data: String, name of .csv data file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        x_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """

    # Read in CSV file
    df1 = pd.read_csv(data, encoding='utf-8').fillna(0) 
    df2 = df1.copy()

    #Configure separate data sets
    df2.drop(df2.index[-1])
    i = df1.shape[0] -1
    while (i > -1):
        # split DataFrame into 2 data sets
        if ((i/3).is_integer()):
            df1 = df1.drop(df1.index[i])  # train set (2/3)
        else:
            df2 = df2.drop(df2.index[i])  # test set (1/3)
        i -=1
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
      
    #Reshapes the DataFrames to only take values from V00 to V95
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1.loc[:,'V00':'V95'].values.reshape(-1, 1))
    flow1 = scaler.transform(df1.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]

   
    #Initialise the data sets as empty
    train, test = [], []   
    #Convert the data from 5 min intervals to 1 hour intervals 
    for i in range(lags, len(flow1)):   
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    #Convert data sets into NumPy arrays for shuffle
    train = np.array(train)
    test = np.array(test)

    #Shuffling the training data to prevent the model from learning any patterns based on the order of the data
    np.random.shuffle(train)

    # Split the training and testing data into input (X) and target (y) variables
    x_train = train[:, :-1]  # All columns except the last one
    y_train = train[:, -1]   # The last column
    x_test = test[:, :-1]    # All columns except the last one
    y_test = test[:, -1]     # The last column


    return x_train, y_train, x_test, y_test, scaler
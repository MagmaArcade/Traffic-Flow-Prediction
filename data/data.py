"""
Processing the data
"""
#import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression



def get_coords(data, scats, junction):
    # Read data file
    df = pd.read_csv(data, encoding='utf-8').fillna(0)

    # Filter the DataFrame around the SCATS, making a new DataFrame that just has this SCAT
    filtered_df = df[df['SCATS Number'] == int(scats)]

    # Remove duplicates if there are any
    filtered_df = filtered_df.drop_duplicates(subset=['NB_LATITUDE', 'NB_LONGITUDE'])
    filtered_df = filtered_df.reset_index()
    i = 0
    lat = 0
    long = 0
    #Calculate the aprx centre of all the junctions. This has flaws and isn't perfect for say, a 90 degree turn
    while ((i + 1) < len(filtered_df)):
        lat = lat + float(filtered_df.loc[i,'NB_LATITUDE'])
        long = long + float(filtered_df.loc[i,'NB_LONGITUDE'])
        i += 1
    lat = lat/i
    long = long/i
    safeIndex = -1
    i = 0
    #Calculate the direction the exit to a junction is
    while ((i + 1) < len(filtered_df)):
        tempa = float(filtered_df.loc[i,'NB_LATITUDE']) - lat
        tempo = float(filtered_df.loc[i,'NB_LONGITUDE']) - long
        if ( abs(tempa) > abs(tempo)):
            angle = math.degrees(math.atan(tempo/tempa))
            if (tempa > 0):
                if (angle > 22):
                    if (tempo > 0):
                        if (junction == "NE"):
                            safeIndex = i
                    else:
                        if (junction == "NW"):
                            safeIndex = i
                else:
                    if (junction == "N"):
                        safeIndex = i
            else:
                if (angle > 22):
                    if (tempo > 0):
                        if (junction == "SE"):
                            safeIndex = i
                    else:
                        if (junction == "SW"):
                            safeIndex = i
                else:
                    if (junction == "S"):
                        safeIndex = i
        else:
            angle = math.degrees(math.atan(tempa/tempo))
            if (tempo > 0):
                if (angle > 22):
                    if (tempa > 0):
                        if (junction == "NE"):
                            safeIndex = i
                    else:
                        if (junction == "SE"):
                            safeIndex = i
                else:
                    if (junction == "E"):
                        safeIndex = i
            else:
                if (angle > 22):
                    if (tempa > 0):
                        if (junction == "NW"):
                            safeIndex = i
                    else:
                        if (junction == "SW"):
                            safeIndex = i
                else:
                    if (junction == "E"):
                        safeIndex = i
        i += 1

    if (safeIndex != -1):
        return filtered_df.loc[safeIndex,'NB_LATITUDE'], filtered_df.loc[safeIndex,'NB_LONGITUDE']
    else:
        return -1, -1

def process_data(data, lags):
    """Process data
    Reshape and split train/test data.

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

    #creates a copy of the CSV file for training purposes
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
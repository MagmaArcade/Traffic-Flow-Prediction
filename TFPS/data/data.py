"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

    df1 = pd.read_csv(data, encoding='utf-8').fillna(0) # this is used to read the csv file
    df2 = df1.copy()

    #Couldn't figure out an easy way to split the data into test and train, while also keeping the labels. 
    #It makes 2 identical DataFrames, and then hopefully deletes the test from one, and the train from the other.
    df2.drop(df2.index[-1])
    i = df1.shape[0] -1
    while (i > 0):
        i -=1
        #1/3 of the data becomes test. No reason for that specific number other than Xiaochus example data
        if ((i/3).is_integer()):
            df1.drop(df1.index[i])
        else:
            df2.drop(df2.index[i])
      



    #takes the range of values from V00 to V95, not sure if its ordered with all V00 next to each other or V00-V95 and then the next row
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1.loc[:,'V00':'V95'].values.reshape(-1, 1)) # scales the data so the model can converge faster
    flow1 = scaler.transform(df1.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2.loc[:,'V00':'V95'].values.reshape(-1, 1)).reshape(1, -1)[0]
    


    train, test = [], []    # initialise the data sets as empty
    for i in range(lags, len(flow1)):   
        train.append(flow1[i - lags: i + 1]) # this converts the data from 5 min intervals to 1 hour intervals 
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1] # takes each row of the array, all elements except the last (takes the first 12 element)
    y_train = train[:, -1] # takes each row of the array, only the last element (takes the 13th element)
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler

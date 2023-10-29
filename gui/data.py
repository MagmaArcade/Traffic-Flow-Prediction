import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(data):
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
"""
Processing the data
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression



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


    #For the IDM acceleration we are using Dijkstra's Algorithm taking into account that all distances cant be -
    #To search the shortest distance and path between 2 nodes we are using a graph

    # Set negative values to zero
    flow1 = np.maximum(flow1, 0)
    flow2 = np.maximum(flow2, 0)

    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph based on the flow data
    for i, f1 in enumerate(flow1):
        G.add_node(f"Node1-{i+1}", flow=f1)

    for i, f2 in enumerate(flow2):
        G.add_node(f"Node2-{i+1}", flow=f2)

    # Add edges between nodes
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                weight = np.abs(G.nodes[node1]["flow"] - G.nodes[node2]["flow"])  # Calculate weight based on flow difference
                G.add_edge(node1, node2, weight=weight)

    # Calculate the shortest path and distance
    shortest_path = nx.shortest_path(G, source="Node1-1", target="Node2-5", weight="weight")
    shortest_distance = nx.shortest_path_length(G, source="Node1-1", target="Node2-5", weight="weight")

    # Print the results
    print(f"Shortest Path: {shortest_path}")
    print(f"Shortest Distance: {shortest_distance:.2f}")
    print(shortest_path)
    print(shortest_distance)

    # Visualize the graph
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, "weight")

    nx.draw(G, pos, with_labels=True, node_size=100)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


    return x_train, y_train, x_test, y_test, scaler


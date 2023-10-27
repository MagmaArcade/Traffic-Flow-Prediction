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

    start_node = "Node1"
    goal_node = "Node2"

    #Implementing the a* algorithm
    #---------------------------------------# 
    def a_star_search(G, start_node, goal_node, heuristic_function):
        open_set = []  # Nodes to be explored
        closed_set = set()  # Nodes already explored
        g_score = {node: float('inf') for node in graph.nodes}
        g_score[start_node] = 0
        f_score = {node: float('inf') for node in graph.nodes}
        f_score[start_node] = heuristic_function(start_node, goal_node)

        while open_set:
            current_node = min(open_set, key=lambda node: f_score[node])
            open_set.remove(current_node)

            if current_node == goal_node:
                # Path found, reconstruct and return it
                path = []
                while current_node != start_node:
                    path.insert(0, current_node)
                    current_node = came_from[current_node]
                path.insert(0, start_node)
                return path

            closed_set.add(current_node)

            for neighbor in graph.neighbors(current_node):
                if neighbor in closed_set:
                    continue  # Ignore neighbors that have already been explored

                tentative_g_score = g_score[current_node] + graph[current_node][neighbor].get('weight', 1)

                if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                    # This is the best path until now
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_function(neighbor, goal_node)
                    if neighbor not in open_set:
                        open_set.append(neighbor)

    return None  # No path found


    # Calculate the shortest path
    shortest_path = a_star_search(G, start_node, goal_node, heuristic_function)
   
    # Print the results
    print(f"Shortest Path: {shortest_path}")
    print(shortest_path)

    return x_train, y_train, x_test, y_test, scaler

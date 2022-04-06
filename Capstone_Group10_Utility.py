"""
Capstone Project Group 10

Copyright (c) 2021 -- This is the 2021 Fall B version of the Template
Licensed
Written by Capstone Group 10 members
"""

import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def rd_csv_file(path):
    """
    Fucntion to read the csv file
    :param path: Path of the csv file
    :return: data frame after reading the csv file
    """
    r = requests.get(path)
    with open('file.csv.gz', 'wb') as fo:
        fo.write(r.content)
    return pd.read_csv("file.csv.gz")

def convert_price(var):
    """
    Function to take variable of a dataframe as an argument and convert it to float type
    :param var: Represents the variable being passed to this function
    :return: clean and converted variable
    """
    return var.str.replace("[$, ]", "").astype("float")

def calc_revenue(price, nights):
    """
    Function to calculate revenue using price multiplied by total nights
    :param price: input price variable
    :param nights: input nights variable
    :return: multiplication of price and nights
    """
    return price * nights

def calc_rmse(y, pred):
    """
    Funciton to calculate root mean square error value of a model output
    :param y: imput y_test data
    :param pred: input predicted y data
    """
    mse = mean_squared_error(y, pred)
    return np.sqrt(mse)

def knn_model(X_train,X_test,y_train,k):
    """
    Function to train kNN model using the training data and generating predicted values for y
    :param X_train: input training dataset
    :param X_test: input testing dataset
    :param y_train: input training result
    """
    pipe = make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=k,algorithm='auto'))
    pipe.fit(X_train,y_train)
    return pipe.predict(X_test)

def Linear_Regression_model(X_train,X_test,y_train):
    """
    Function to train kNN model using the training data and generating predicted values for y
    :param X_train: input training dataset
    :param X_test: input testing dataset
    :param y_train: input training result
    """
    lr_pipe = make_pipeline(StandardScaler(), LinearRegression(normalize=True))
    lr_pipe.fit(X_train,y_train)
    return lr_pipe.predict(X_test)

def xGBoost_model(X_train,X_test,y_train,i):
    """
    Function to train kNN model using the training data and generating predicted values for y
    :param X_train: input training dataset
    :param X_test: input testing dataset
    :param y_train: input training result
    :param i: It determines the maximum depth of the individual regression estimators
    """
    gb_pipe  = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42,max_depth = i))
    gb_pipe.fit(X_train,y_train)
    return gb_pipe.predict(X_test)

def Decision_Tree_model(X_train,X_test,y_train,i):
    """
    Function to train kNN model using the training data and generating predicted values for y
    :param X_train: input training dataset
    :param X_test: input testing dataset
    :param y_train: input training result
    :param i: It determines the maximum depth of the tree
    """
    decTree_pipe  = make_pipeline(StandardScaler(), DecisionTreeRegressor(random_state=250,max_depth = i))
    decTree_pipe.fit(X_train,y_train)
    return decTree_pipe.predict(X_test)

def Random_Forest_model(X_train,X_test,y_train,i):
    """
    Function to train kNN model using the training data and generating predicted values for y
    :param X_train: input training dataset
    :param X_test: input testing dataset
    :param y_train: input training result
    :param i: It determines the maximum depth of the tree
    """
    rf_pipe= make_pipeline(StandardScaler(),RandomForestRegressor(random_state=44,max_depth = i))
    rf_pipe.fit(X_train,y_train)
    return rf_pipe.predict(X_test)

if __name__ == '__main__':
    print('Initializing')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd

def load_train_dev_test(data_path, random_state=42):
    """
    Load the data and split it into train, dev and test sets
    """
    
    dataset = pd.read_csv(data_path)
    X = dataset.drop(columns=['ClosePrice'])
    y = dataset['ClosePrice']

    # TODO: A split of the data by publish date could be a better approach, but I will use a random split for now
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, train_size=0.9, random_state=random_state)

    return (X_train, y_train), (X_dev, y_dev)

def rmse(y_true, y_pred):
    """
    Returns the root mean squared
    """
    return mean_squared_error(y_true, y_pred) ** 0.5
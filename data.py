import cudf
import os
import pandas as pd
from pathlib import Path


def load_holdout(columns=None):
    """ Loads the holdout data held in Application_Holdout.csv as a cudf DataFrame. If a feather file does not exists it
    creates one to load from next time. Otherwise it loads from the feather file for faster load.
    
    Parameters
    ----------
    columns (list, default=None) : a list of columns to import. If None, all columns are loaded

    Returns
    -------
    DataFrame : the holdout data    
    """
    p = Path('./data/Application_Holdout.feather')
    if p.exists():
        data = pd.read_feather(p._str, columns=columns)
    else:
        data = pd.read_csv('./data/Application_Holdout.csv', usecols=columns)
        data.to_feather(p._str)
    return cudf.from_pandas(data)


def load_train(columns=None):
    """ Loads the training data held in Application_Train.csv as a cudf DataFrame. If a feather file does not exists it 
    creates one to load from next time. Otherwise it loads from the feather file for faster load. 
    
    Parameters
    ----------
    columns (list, default=None) : a list of columns to import. If None, all columns are loaded

    Returns
    -------
    DataFrame : the training data    
    """
    p = Path('./data/Application_Train.feather')
    if p.exists():
        data = pd.read_feather(p._str, columns=columns)
    else:
        data = pd.read_csv('./data/Application_Train.csv', usecols=columns)
        data.to_feather(p._str)
    return cudf.from_pandas(data)


def one_hot_encode(data, columns=["BatterHand", "PitcherHand", "DayNight", "PitchType", "TopInning"]):
    """ One hot encodes the given categorical columns 
    
    Parameters
    ----------
    data (DataFrame) : the data to one hot encode

    columns (list, default=["BatterHand", "PitcherHand", "DayNight", "PitchType", "TopInning"]) : the columns to one hot 
    encode

    Returns
    -------
    DataFrame : the data with the specified columns one hot encoded
    """
    return cudf.get_dummies(data, columns=columns)


def make_label(data):
    """ Makes the label column as 1 if the pitch result was "Swinging strike" and 0 otherwise 
    
    Parameters
    ----------
    data (DataFrame) : the data containing the column "PitchResult"

    Returns
    -------
    DataFrame : the data with a column "SwingAndMiss"    
    """
    new_data = cudf.get_dummies(data, columns=["PitchResult"])
    new_cols = set(new_data.columns) - set(data.columns)
    new_data["SwingAndMiss"] = new_data["PitchResult_Swinging strike"]
    return new_data.drop(columns=list(new_cols))

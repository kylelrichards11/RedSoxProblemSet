from datetime import datetime

import cudf
from numba import cuda
import numpy as np
import pandas as pd


def _group_by_unique_count(group_col):
    """ Groups by the unique values in the given column and returns a series for the count of each unique value in the 
    order they are seen in the group_col
    
    Parameters
    ----------
    group_col (Series) : the column to group by. Contains group by keys and unique keys separated by a comma

    Returns
    -------
    list : the counts        
    """
    mapping = {}
    d = {}
    last_index = {}
    for uq in np.sort(group_col.unique().to_array()):
        group_by_key, uq_key = uq.split(',')
        if group_by_key not in d:
            d[group_by_key] = {uq_key: 0}
            last_index[group_by_key] = 1
        elif uq_key not in d[group_by_key]:
            d[group_by_key][uq_key] = last_index[group_by_key]
            last_index[group_by_key] += 1
        mapping[uq] = d[group_by_key][uq_key]
    return list(map(lambda x: mapping[x], list(group_col.to_array())))


def appearance_gap(data):
    """ Calculates the number of days between the pitcher's current appearance and their last appearance. The gap is 
    capped at 14 days 
    
    Parameters
    ----------
    data (DataFrame) : the dataset containing the pitches

    Returns
    -------
    DataFrame : the dataset with the feature AppearanceGap
    """
    group_col = (
        data["PitcherID"].astype(str) +
        ',' +
        data["GameDate"].astype(str) +
        '_' +
        data["DayNight_Night"].astype(str)
    )
    mapping = {}
    d = {}
    last_date = {}
    for uq in np.sort(group_col.unique().to_array()):
        pitcher_key, date_key = uq.split(',')
        date = datetime.strptime(date_key.split('_')[0], "%Y-%m-%d")
        if pitcher_key not in d:
            d[pitcher_key] = {date_key: 14}
            last_date[pitcher_key] = date
        elif date_key not in d[pitcher_key]:
            d[pitcher_key][date_key] = min(14, (date - last_date[pitcher_key]).days)
            last_date[pitcher_key] = date
        mapping[uq] = d[pitcher_key][date_key]
    data["AppearanceGap"] = list(map(lambda x: mapping[x], list(group_col.to_array())))
    return data


def pa_of_game(data):
    """ Calculates a numeric value that gives the current number of batters the pitcher has thrown against in the 
    current game.

    Parameters
    ----------
    data (DataFrame) : the dataset containing the pitches

    Returns
    -------
    DataFrame : the dataset with the feature PAOfGame
    """
    group_col = (
        data["PitcherID"].astype(str) +
        data["GameNumber"].astype(str) +
        ',' +
        data["Inning"].astype(str).str.zfill(2) +
        data["PAOfInning"].astype(str).str.zfill(3)
    )
    data["PAOfGame"] = _group_by_unique_count(group_col)
    return data


def pa_of_season(data):
    """ Calculates a numeric value that gives the current number of batters the pitcher has thrown against in the 
    current season.

    Parameters
    ----------
    data (DataFrame) : the dataset containing the pitches

    Returns
    -------
    DataFrame : the dataset with the feature PAOfSeason
    """
    group_col = (
        data["PitcherID"].astype(str) +
        data["Season"].astype(str) +
        ',' +
        data["GameDate"].astype(str) +
        data["DayNight_Night"].astype(str) +
        data["GameNumber"].astype(str) +
        data["Inning"].astype(str).str.zfill(2) +
        data["PAOfInning"].astype(str).str.zfill(3)
    )
    data["PAOfSeason"] = _group_by_unique_count(group_col)
    return data


def pitch_of_game(data):
    """ Calculates a numeric value that gives the current number of pitches the pitcher has thrown in the current game.

    Parameters
    ----------
    data (DataFrame) : the dataset containing the pitches

    Returns
    -------
    DataFrame : the dataset with the feature PitchOfGame
    """
    def count(Index, PitchOfGame):
        for i in range(cuda.threadIdx.x, len(Index), cuda.blockDim.x):
            PitchOfGame[i] = i

    results = data.groupby(["PitcherID", "GameNumber"]).apply_grouped(
        count,
        incols=["Index"],
        outcols={"PitchOfGame": np.int16}
    )
    return results.sort_index()


def pitch_of_season(data):
    """ Calculates a numeric value that gives the current number of pitches the pitcher has thrown in the current 
    season.

    Parameters
    ----------
    data (DataFrame) : the dataset containing the pitches

    Returns
    -------
    DataFrame : the dataset with the feature PitchOfSeason
    """
    def count(Index, PitchOfSeason):
        for i in range(cuda.threadIdx.x, len(Index), cuda.blockDim.x):
            PitchOfSeason[i] = i

    results = data.groupby(["PitcherID", "Season"]).apply_grouped(
        count,
        incols=["Index"],
        outcols={"PitchOfSeason": np.int16}
    )
    return results.sort_index()


def runs_ahead(data):
    """ Calulcates the number of runs the pitcher is currently ahead by. If the pitcher is behind then the number is 
    negative.

    Parameters
    ----------
    data (DataFrame) : the dataset containing the pitches

    Returns
    -------
    DataFrame : the dataset with the feature RunsAhead
    """
    def check_negate(ScoreDiff, TopInning_TOP, RunsAhead):
        for i, (s, t) in enumerate(zip(ScoreDiff, TopInning_TOP)):
            RunsAhead[i] = s if t == 0 else -s

    data["ScoreDiff"] = data["AwayScore"] - data["HomeScore"]
    data = data.apply_rows(check_negate, incols=["ScoreDiff", "TopInning_TOP"],
                           outcols={"RunsAhead": np.int16}, kwargs={})
    data = data.drop(columns=["ScoreDiff"])
    return data

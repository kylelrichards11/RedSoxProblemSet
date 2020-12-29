import cudf
from numba import cuda
import numpy as np
import pandas as pd

# TODO:
# PA of game
# PA of season
# Pitch of game
# Pitch of season
# Pitch runs ahead


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

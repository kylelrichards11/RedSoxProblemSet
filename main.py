from data import load_holdout, load_train, make_label, one_hot_encode

if __name__ == "__main__":

    # Only load columns in both datasets (except for PitchResult for training)
    cols = [
        "AwayScore",
        "B1",
        "B2",
        "B3",
        "Balls",
        "BatterHand",
        "DayNight",
        "GameDate",
        "GameNumber",
        "GameSeqNum",
        "HomeScore",
        "Inning",
        "Outs",
        "PAOfInning",
        "PitchBreakHorz",
        "PitchBreakVert",
        "PitchOfPA",
        "PitchResult",
        "PitchType",
        "PitcherHand",
        "PitcherID",
        "PlateLocX",
        "PlateLocZ",
        "ReleaseLocX",
        "ReleaseLocY",
        "ReleaseLocZ",
        "ReleaseSpeed",
        "ReleaseVelocityX",
        "ReleaseVelocityY",
        "ReleaseVelocityZ",
        "Season",
        "SpinRate",
        "Strikes",
        "TopInning",
    ]

    data = load_train(columns=cols)
    data = one_hot_encode(data)
    data = make_label(data)
    print(data)

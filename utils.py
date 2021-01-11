import json


def print_time(stime, etime, prefix="Finished in "):
    """ Prints time in appropriate units with labels

    Parameters
    ----------
    stime (float) : start time in seconds

    etime (float) : end time in seconds

    prefix (string), default='Finished in ' : what to print before the time

    Returns
    -------
    None
    """
    diff = etime - stime
    if diff < 100:
        print(f"{prefix}{diff:.2f} seconds")
    elif diff < 6000:
        print(f"{prefix}{diff/60:.2f} minutes")
    else:
        print(f"{prefix}{diff/3600:.2f} hours")


def write_dict(d, filename):
    """ Writes the given dictionary to the given filename as a json file 
    
    Parameters
    ----------
    d (dict) : the dictionary to write (also works with lists)

    filename (str) : the filename to write to

    Returns
    -------
    None    
    """
    with open(f"{filename}.json", 'w') as f:
        json.dump(d, f)

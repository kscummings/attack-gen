import os
import json

from pathlib import Path

def get_data_path():
    """
    Return path stored in transit-alliance root directory
    """
    # keep everything before and up to 'transit-alliance'
    thisdir=Path(__file__).parent.absolute()
    path = os.path.join(thisdir, # utils
            "..", # src
            "..", # transit-alliance
            'data-path.txt')

    # Read first line.
    with open(path, 'r') as f:
        return f.readline().strip()

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


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

def get_xwalk():
    """
    Read in sensor sets
    """
    filepath=os.path.join(get_data_path,"sensor_xwalk.csv")

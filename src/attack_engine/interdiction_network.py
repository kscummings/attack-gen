"""
Build the network to run interdiction model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import wntr

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


sys.path.append('..')
from src.utils import get_data_path
from src.preprocess import WaterNetwork

class InterdictionNetwork:
    """
    Network considered by attacker
    """
    def __init__(self, filepath):
        self.original_wn=WaterNetwork(filepath)
        self.dem=self.original_wn.sim_demand()

        G=self.original_wn.get_network()

        # build the network
        ### simulate demand to add to each node
        ### assume supply is equal to total simulated demand (can satisfy all)
        ### add as network attribute
        ### add edges:
        ##### reservoir to all tanks
        ##### all demand nodes to artificial sink

        self.topo=G

    def reduce_demand(self,null_nodes):
        """
        update demand (property 'dem') of each node in the network
        ### Keyword Arguments
        *`null_nodes` - zero out demand of the nodes in this list
        """
        dem=nx.get_node_attributes(self.topo,'dem')

        # update all of the demands in the network
        nx.set_node_attributes(self.topo,dem)

    def update_demand(self,dem):
        """
        set 'dem' attribute to specific demand set
        """
        self.dem=dem

        # reset in the network too

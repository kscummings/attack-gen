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

        # water network
        self.original_wn=WaterNetwork(filepath)

        # full week of simulated demand
        self.dem=self.original_wn.sim_demand()

        # just the demands
        tanks,reservoirs,junctions=self.original_wn.get_nodes()
        self.dem=self.dem.drop(np.hstack((tanks,reservoirs)),axis=1)
        totaldem=self.dem.sum(axis=0)

        # build network
        G=self.original_wn.get_network()

        # name source, build reservoir's out-edges to tanks
        assert ["R1"]==reservoirs
        assert all([len(G[u][v])==1 for (u,v,_) in G.edges])
        G.add_node("source")
        G.add_edge("source","R1","P0") # source to reservoir
        source_edges=[("R1",tank,"P0") for tank in tanks]
        G.add_edges_from(source_edges)

        # build sink and its in-edges from junctions with demand
        G.add_node("sink")
        sink_edges=[(junction,"sink","P0") for junction in junctions]

        # edge attributes: demand
        # assumes supply can satisfy all
        totaldem_dict={(j,s,p):totaldem[j] for (j,s,p) in sink_edges}
        totaldem_dict.update({("source","R1","P0"):-sum(totaldem_dict.values())})
        totaldem_dict.update({e : 0 for e in G.edges if (e not in source_edges) & (e not in sink_edges)})
        nx.set_edge_attributes(G,totaldem_dict,"demand")

        # finalize
        self.topo=G
        self.source_name="source"
        self.sink_name="sink"


    def reduce_demand(self,null_nodes):
        """
        update demand (property 'dem') of each node in the network
        ### Keyword Arguments
        *`null_nodes` - zero out demand of the nodes in this list
        """
        dem=nx.get_edge_attributes(self.topo,'demand')

        # update all of the demands in the network
        nx.set_node_attributes(self.topo,dem)

    def update_demand(self,dem):
        """
        set 'dem' attribute to specific demand set
        """
        self.dem=dem

        # reset in the network too

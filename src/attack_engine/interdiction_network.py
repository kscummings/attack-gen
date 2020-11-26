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

from functools import reduce


sys.path.append('..')
from src.utils import get_data_path
from src.preprocess import WaterNetwork

class InterdictionNetwork:
    """
    Auxiliary max-flow network
    """
    def __init__(self, filepath):
        """
        ### Keyword Arguments
        *`filepath` - location of water network input file
        """

        # water network
        self.original_wn=WaterNetwork(filepath)

        # full week of simulated demand
        tanks,reservoirs,junctions=self.original_wn.get_nodes()
        assert ["R1"]==reservoirs
        dem=self.original_wn.sim_demand()

        # build network
        self.source_name="source"
        self.sink_name="sink"
        G_original=self.original_wn.get_network()

        # name source, build reservoir's out-edges to tanks
        G.add_node(self.source_name)
        G.add_edge(self.source_name,"R1") # source to reservoir
        self.source_edges=[("R1",tank) for tank in tanks]
        G.add_edges_from(self.source_edges)

        # build sink and its in-edges from junctions with demand
        G.add_node(self.sink_name)
        self.sink_edges=[(junction,self.sink_name) for junction in junctions]

        # finalize
        self.G=G
        self.update_demand(dem)



    def reduce_demand(self,null_nodes):
        """
        update demand (property 'dem') of each node in the network
        ### Keyword Arguments
        *`null_nodes` - zero out demand of the nodes in this list
        """
        dem=nx.get_edge_attributes(self.topo,'demand')
        dem.update({j:0 for j in null_nodes})
        nx.set_edge_attributes(self.topo,dem)

    def update_demand(self,dem):
        """
        set 'dem' attribute to specific demand set
        update in network: demand, tank capacities
        """
        tanks,reservoirs,junctions=self.original_wn.get_nodes()
        supply=dem.drop(np.hstack((junctions,reservoirs)),axis=1)
        dem=dem.drop(np.hstack((tanks,reservoirs)),axis=1)
        self.dem=dem

        # assumes supply can satisfy all
        totaldem=self.dem.sum(axis=0)
        totaldem_dict={(j,s):totaldem[j] for (j,s) in self.sink_edges}
        totaldem_dict.update({(self.source_name,"R1"):-sum(totaldem_dict.values())})
        totaldem_dict.update({e:0 for e in self.G.edges if (e not in self.source_edges) & (e not in self.sink_edges)})
        nx.set_edge_attributes(self.G,totaldem_dict,"demand")

        # compute capacities of tanks: the amount that it gave throughout the week
        cap=supply.agg([sum_neg])
        cap_dict={(u,v):cap[v].sum_neg for (u,v) in self.source_edges}
        cap_dict.update({(u,v):M for (u,v) in self.G.edges if (u,v) not in self.source_edges})
        nx.set_edge_attributes(self.G,cap_dict,"capacity")


def sum_neg(series):
    return reduce(lambda a,b: a+b if (b<0) else a, series)

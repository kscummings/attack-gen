from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import wntr

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from os import path

sys.path.append('..')
from src.utils import get_data_path


class WaterNetwork:
    """
    obtain ctown water network
    adapted from https://github.com/4flixt/2019_WNTR_Surrogate_Model
    """

    def __init__(self, filePath):
        """
        constructor
        ### Keyword Arguments
        *`filepath` - string
        """
        self.__filePath = filePath
        self.wn = wntr.network.WaterNetworkModel(self.__filePath)

    def get_nodes(self):
        """
        get all nodes: tanks, reservoirs, junctions
        """
        node_names = self.wn.node_name_list
        tank_names = [tank for tank in node_names if tank.startswith("T")]
        reservoir_names = [reservoir for reservoir in node_names if reservoir.startswith("R")]
        junction_names = [junction for junction in node_names if junction.startswith("J")]
        return [tank_names, reservoir_names, junction_names]

    def get_links(self):
        """
        get all links: pumps, pipes, valves
        """
        link_names = self.wn.link_name_list
        pump_names = [pump for pump in link_names if pump.startswith("PU")]
        pipe_names = [pipe for pipe in link_names if pipe.startswith("P") and not(pipe.startswith("PU"))]
        valve_names = [valve for valve in link_names if valve.startswith("V")]

        return [pump_names, pipe_names, valve_names]

    def get_network(self):
        """
        topology of water network
        """
        return self.wn.get_graph()

    def sim_demand(self):
        """
        simulate a week of demand on this water network
        """
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim()
        return results.node['demand']


def main():
    FILEPATH=path.join(get_data_path(),"CTOWN.INP")
    wn=WaterNetwork(FILEPATH)
    dem=wn.sim_demand()
    tanks,reservoirs,junctions=wn.get_nodes()
    res=dem.drop(np.hstack((junctions,tanks)),axis=1)
    supply=dem.drop(np.hstack((junctions,reservoirs)),axis=1)
    dem=dem.drop(np.hstack((tanks,reservoirs)),axis=1)

    # look at time series
    res.plot()
    supply.plot()
    plt.show()
    dem.plot()
    plt.show()

    # average supply and demand
    plt.hist(supply.mean(axis=0),density=True,bins=30)
    plt.show()
    plt.hist(dem.mean(axis=0),density=True,bins=30)
    plt.show()


if __name__ == '__main__':
    main()

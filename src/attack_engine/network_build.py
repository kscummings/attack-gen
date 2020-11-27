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
        self.wn.options.hydraulic.demand_model = 'DD'

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
        G_original=self.wn.get_graph()

        # single pipe per node pair
        assert all([len(G_original[u][v])==1 for (u,v,_) in G_original.edges])

        G=nx.DiGraph()

        # same nodes, same data
        for node, node_data in G_original.nodes(data=True):
            G.add_node(node,pos=node_data['pos'],type=node_data['type'])

        # updated edges, same data
        # move pipe/pump/valve name to edge attribute
        for (u,v,id) in G_original.edges:
            G.add_edge(u,v,edge_id=id)
        for (u,v,edge_data) in G_original.edges(data=True):
            G[u][v]['type']=edge_data['type']

        return G


    def sim_demand(self):
        """
        simulate a week of demand on this water network
        units: m^3/s
        """
        sim = wntr.sim.EpanetSimulator(self.wn)
        results = sim.run_sim()
        return results.node['demand']


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

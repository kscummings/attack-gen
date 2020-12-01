from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import sys
import wntr

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from os import path

sys.path.append('..')
from src.utils import get_data_path

CC_EDGES=[("J302","J307"),("J332","J301"),("J288","J300"),("J422","J420")]

# big ole number
M=1e7

RESERVOIR="R1"


########################CONSTRUCTORS

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

        # get link lengths, add as edge distance attribute
        pipe_len={pipe:pd.length for pipe,pd in self.wn.pipes()}
        mean_pipe_len=np.mean(list(pipe_len.values()))

        # updated edges, same data
        # move pipe/pump/valve name to edge attribute
        for (u,v,id) in G_original.edges:
            G.add_edge(u,v,edge_id=id)
            if id in pipe_len.keys():
                G[u][v]['distance']=pipe_len[id]
            else:
                G[u][v]['distance']=mean_pipe_len
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
    * adds source and sink
    * connects source to reservoir
    * connects all demand nodes to sink
    * connects reservoir to all tanks
    * converts network (not auxiliary) to being undirected
    * characterizes demand as simulated demand over a week
    * adds capacities to tank edges equivalent to supplied demand
    """
    def __init__(self, filepath):
        """
        ### Keyword Arguments
        *`filepath` - location of water network input file
        ### Attributes
        *`original_wn`
        *`G`
        *`source_name`
        *`sink_name`
        *`reservoir`
        *`source_edges`
        *`sink_edges`
        *`fortified_edges`
        *`original_dem`
        """
        ### initialize
        # water network
        self.original_wn=WaterNetwork(filepath)

        # full week of simulated demand
        tanks,reservoirs,junctions=self.original_wn.get_nodes()
        dem=self.original_wn.sim_demand()

        # build network
        self.source_name="source"
        self.sink_name="sink"
        self.reservoir=RESERVOIR
        G=self.original_wn.get_network()
        assert [self.reservoir]==reservoirs

        ### build fortified edges (not interdictable)
        # fortify reservoir -> tank edges in original network
        H=G.to_undirected()
        path_lengths,paths=nx.single_source_dijkstra(H,self.reservoir,weight="distance")
        tank_paths=[get_epath(paths[tank]) for tank in tanks]
        self.fortified_edges=pd.unique([edge for tank_path in tank_paths for edge in tank_path])

        ### build auxiliary graph
        # convert original graph to undirected graph
        assert not any([(v,u) in G.edges for (u,v) in G.edges])
        backwards=[(v,u,ed) for (u,v,ed) in G.edges(data=True)]
        G.add_edges_from(backwards)

        # entire auxiliary graph is directed
        # name source, build reservoir's out-edges to tanks
        G.add_node(self.source_name)
        G.add_edge(self.source_name,self.reservoir) # source to reservoir
        self.source_edges=[(self.reservoir,tank) for tank in tanks]
        G.add_edges_from(self.source_edges)

        # build sink and its in-edges from junctions with demand
        G.add_node(self.sink_name)
        self.sink_edges=[(junction,self.sink_name) for junction in junctions]
        G.add_edges_from(self.sink_edges)

        # sink->source edge
        G.add_edge(self.sink_name,self.source_name)

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

    def update_flows(self,dem):
        """
        reset flows through network
        update in network: demand, tank capacities
        """
        tanks,reservoirs,junctions=self.original_wn.get_nodes()
        supply=dem.drop(np.hstack((junctions,reservoirs)),axis=1)
        dem=dem.drop(np.hstack((tanks,reservoirs)),axis=1)

        # assumes supply can satisfy all
        totaldem=dem.sum(axis=0)
        totaldem_dict={(j,s):totaldem[j] for (j,s) in self.sink_edges}
        totaldem_dict.update({(self.source_name,self.reservoir):-sum(totaldem_dict.values())})
        totaldem_dict.update({e:0 for e in self.G.edges if (e not in self.source_edges) & (e not in self.sink_edges)})
        nx.set_edge_attributes(self.G,totaldem_dict,"demand")
        self.original_dem=totaldem_dict

        # compute capacities of tanks: the amount that it gave throughout the week
        cap=supply.agg([sum_neg])
        cap_dict={(u,v):cap[v].sum_neg for (u,v) in self.source_edges}
        cap_dict.update({(u,v):M for (u,v) in self.G.edges if (u,v) not in self.source_edges})
        nx.set_edge_attributes(self.G,cap_dict,"capacity")



########################HELPERS



def edge_sensor_xwalk(G,data_path):
    """
    Determine ted components corresponding to each sensor set to build edge/sensor xwalk
    (note this might not work anymore post-undirected but doesnt matter bc already built it)
    ### Keyword Arguments
    *`G` - water network graph (not auxiliary graph)
    *`data_path` - directory containing system xwalk
    """
    sensor_xwalk=pd.read_csv(os.path.join(data_path,"sensor_xwalk.csv"))
    tankcol=[col for col in sensor_xwalk if col.startswith('T')]
    assert all(sensor_xwalk[tankcol].sum(axis=0)==1)
    sensor_switch={}
    for i in range(len(sensor_xwalk)):
        sensor_set_id=sensor_xwalk.at[i,"sensor_set_id"]
        tanks=[tank for tank in tankcol if sensor_xwalk.at[i,tank]==1]
        sensor_switch.update({tank:sensor_set_id for tank in tanks})

    # remove CC edges
    G.remove_edges_from(CC_EDGES)

    # find connected components
    edge_sensor_dict={}
    for cc in nx.weakly_connected_components(G):
        cc_edge_set=G.subgraph(cc).edges
        for tank in tankcol:
            if tank in cc:
                edge_sensor_dict.update({edge:sensor_switch[tank] for edge in cc_edge_set})

    # manually add removed edges
    edge_sensor_dict.update({
        ("J288","J300"):sensor_switch["T3"],
        ("J302","J307"):sensor_switch["T6"],
        ("J332","J301"):sensor_switch["T5"],
        ("J422","J420"):sensor_switch["T2"]
    })

    # write the dictionary
    with open(path.join(data_path,"edge_sensor_xwalk.csv"), 'w', newline='') as csvfile:
        fieldnames=['origin','dest','sensor_set_id']
        writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (u,v),sensor_id in edge_sensor_dict.items():
            writer.writerow({'origin':u,'dest':v,'sensor_set_id':sensor_id})

    # restore graph
    G.add_edges_from(CC_EDGES)

def get_epath(npath):
    plen=len(npath)
    return [(npath[i],npath[i+1]) for i in range(plen-1)]

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
    # res.plot()
    # supply.plot()
    # plt.show()
    # dem.plot()
    # plt.show()
    #
    # # average supply and demand
    # plt.hist(supply.mean(axis=0),density=True,bins=30)
    # plt.show()
    # plt.hist(dem.mean(axis=0),density=True,bins=30)
    # plt.show()


if __name__ == '__main__':
    main()

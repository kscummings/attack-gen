from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import sys
import wntr

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from copy import copy
from math import pi
from networkx.algorithms import bfs_tree
from os import path

sys.path.append('..')
from src.utils import get_data_path

### constants
CC_EDGES=[("J302","J307"),("J332","J301"),("J288","J300"),("J422","J420")]
NETWORK_COL=['edge_id','origin','dest','dem','capacity','source','sink','fortified']
WATER_VELOCITY=2.4 #m/s
M=1e7 # big ole number
RESERVOIR="R1"
TIME_INCREMENT=900 # simulation increment (seconds) = 15 mins
TIME_HORIZON=604800 # simulation length - 7 weeks


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
        pipe_len={pipe:pdat.length for pipe,pdat in self.wn.pipes()}
        pipe_diam={pipe:pdat.diameter for pipe,pdat in self.wn.pipes()}
        mean_pipe_len=np.mean(list(pipe_len.values()))
        mean_diam=np.mean(list(pipe_diam.values()))

        # updated edges, same data
        # move pipe/pump/valve name to edge attribute
        for (u,v,id) in G_original.edges:
            G.add_edge(u,v,edge_id=id)
            if id in pipe_len.keys():
                G[u][v]['distance']=pipe_len[id]
                G[u][v]['diameter']=pipe_diam[id]
            else:
                G[u][v]['distance']=mean_pipe_len
                G[u][v]['diameter']=mean_diam

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
        return results



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
        tanks,reservoirs,junctions=self.original_wn.get_nodes()

        # build network
        self.source_name="source"
        self.sink_name="sink"
        self.reservoir=RESERVOIR
        G=self.original_wn.get_network()
        assert [self.reservoir]==reservoirs

        ### build fortified edges (not interdictable) - backwards and forwards
        # fortify reservoir -> tank edges in original network
        H=G.to_undirected()
        path_lengths,paths=nx.single_source_dijkstra(H,self.reservoir,weight="distance")
        tank_paths=[get_epath(paths[tank]) for tank in tanks]
        self.fortified_edges=list(pd.unique([edge for tank_path in tank_paths for edge in tank_path]))
        self.fortified_edges=self.fortified_edges+[(v,u) for (u,v) in self.fortified_edges]
        self.fortified_edges=list(pd.unique(self.fortified_edges))

        ### build auxiliary graph
        # convert original graph to undirected graph
        assert not any([(v,u) in G.edges for (u,v) in G.edges])
        backwards=[]
        for (u,v,ed) in G.edges(data=True):
            ed_rev=copy(ed)
            if 'edge_id' in ed.keys():
                ed_rev['edge_id']="%s_rev"%ed['edge_id'] # unique edge id
            backwards.append((v,u,ed_rev))
        G.add_edges_from(backwards)

        # entire auxiliary graph is directed
        # name source, build reservoir's out-edges to tanks
        G.add_node(self.source_name)
        G.add_edge(self.source_name,self.reservoir) # source to reservoir
        self.source_edges=[(self.reservoir,tank) for tank in tanks]
        self.source_edges.append((self.source_name,self.reservoir))
        G.add_edges_from(self.source_edges)

        # build sink and its in-edges from junctions with demand
        G.add_node(self.sink_name)
        self.sink_edges=[(junction,self.sink_name) for junction in junctions]
        G.add_edges_from(self.sink_edges)

        # sink->source edge
        G.add_edge(self.sink_name,self.source_name)

        # finalize
        self.G=G

        # build demands and capacities
        self.sim()


    def sim(self):
        """
        simulate new world over network and update network
        """
        results=self.original_wn.sim_demand()
        self.update_flows(results)

    def update_flows(self,results):
        """
        reset flows through network, given simulated demand
        update in network: demand, tank and pipe capacities (all edge cap)
        """
        dem=results.node['demand']
        tanks,reservoirs,junctions=self.original_wn.get_nodes()
        _,pipes,_=self.original_wn.get_links()
        supply=dem.drop(np.hstack((junctions,reservoirs)),axis=1)
        dem=dem.drop(np.hstack((tanks,reservoirs)),axis=1)

        # build demand
        # assumes supply can satisfy all
        totaldem=dem.sum(axis=0)*TIME_INCREMENT
        totaldem_dict={(j,t):totaldem[j] for (j,t) in self.sink_edges}
        nx.set_edge_attributes(self.G,totaldem_dict,"demand")
        self.original_dem=totaldem_dict

        # capacities on tanks (total vol over time horizon)
        inflow=supply.max()
        outflow=abs(supply.min())
        tank_cap=TIME_HORIZON/(1/inflow+1/outflow)

        # capacities on pipes (total vol over time horizon)
        vel=results.link['velocity'].max()
        pipe_cap={}
        for u,v,ed in self.G.edges(data=True):
            if 'edge_id' in ed.keys():
                edge_id=ed['edge_id'].replace("_rev",'')
                if edge_id in pipes:
                    pipe_cap[(u,v)]=pi*ed['diameter']**2*vel[edge_id]*TIME_HORIZON/4

        # build capacity dict
        capdict=pipe_cap
        capdict.update({(r,t):tank_cap[t] for (r,t) in self.source_edges if t!=self.reservoir})
        nx.set_edge_attributes(self.G,capdict,"capacity")

    def lights_off(self,null_nodes):
        """
        deactivate demands in given set of nodes
        ### Keyword Arguments
        *`null_nodes` - zero out demand of the nodes in this list
        """
        dem=nx.get_edge_attributes(self.G,"demand")
        dem.update({(j,self.sink_name):0 for j in null_nodes})
        nx.set_edge_attributes(self.G,dem,"demand")

    def lights_on(self):
        """
        ensure all demands activated
        """
        nx.set_edge_attributes(self.G,self.original_dem,"demand")

    def get_cap(self):
        """
        return capacity attribute
        """
        return nx.get_edge_attributes(self.G,"capacity")

    def get_demand(self):
        """
        return demand attribute
        """
        return nx.get_edge_attributes(self.G,"demand")

    def to_csv(self,filepath):
        """
        record information necessary to build interdiction network in julia
        ### Writes
        *`edge.csv` - origin, destination, demand, source_edge (1-0), sink_edge (1-0), fortified_edge (1-0)
        """
        dat=pd.DataFrame(index=range(len(self.G.edges)),columns=NETWORK_COL)
        edge_ids,origin,dest,demand,source,sink,fortified,capacity=[],[],[],[],[],[],[],[]
        t=0
        for u,v,data in self.G.edges(data=True):
            t+=1
            # contingent characteristics
            edge_id = data['edge_id'] if 'edge_id' in data.keys() else "edge"+str(t)
            dem = data['demand'] if 'demand' in data.keys() else ""
            cap = data['capacity'] if 'capacity' in data.keys() else M

            edge_ids.append(edge_id)
            origin.append(u)
            dest.append(v)
            demand.append(dem)
            source.append((u,v) in self.source_edges)
            sink.append((u,v) in self.sink_edges)
            fortified.append((u,v) in self.fortified_edges)
            capacity.append(cap)

        # record and return
        dat['edge_id']=edge_ids
        dat['origin']=origin
        dat['dest']=dest
        dat['dem']=demand
        dat['source']=source
        dat['sink']=sink
        dat['fortified']=fortified
        dat['capacity']=capacity
        dat.to_csv(filepath)

        return dat



########################HELPERS



def edge_sensor_xwalk(intnet,data_path):
    """
    Determine ted components corresponding to each sensor set to build edge/sensor xwalk
    (note this might not work anymore post-undirected but doesnt matter bc already built it)
    ### Keyword Arguments
    *`intnet` - interdiction network
    *`data_path` - directory containing system xwalk
    """
    # revert from auxiliary
    G=copy(intnet.G)
    G.remove_edges_from(intnet.sink_edges)
    G.remove_edges_from(intnet.source_edges)
    G.remove_edges_from([(intnet.sink_name,intnet.source_name)])

    sensor_xwalk=pd.read_csv(os.path.join(data_path,"sensor_xwalk.csv"))
    tankcol=[col for col in sensor_xwalk if col.startswith('T')]
    assert all(sensor_xwalk[tankcol].sum(axis=0)==1)
    sensor_switch={}
    for i in range(len(sensor_xwalk)):
        sensor_set_id=sensor_xwalk.at[i,"sensor_set_id"]
        tanks=[tank for tank in tankcol if sensor_xwalk.at[i,tank]==1]
        sensor_switch.update({tank:sensor_set_id for tank in tanks})

    # remove CC edges, backwards and forwards
    forward=[(u,v,G.get_edge_data(u,v)) for (u,v) in CC_EDGES]
    backward=[(v,u,G.get_edge_data(v,u)) for (u,v) in CC_EDGES]
    G.remove_edges_from(forward)
    G.remove_edges_from(backward)

    # find connected components
    edge_sensor_dict={}
    for cc in nx.weakly_connected_components(G):
        G_sub=G.subgraph(cc)
        cc_edge_ids=nx.get_edge_attributes(G_sub,"edge_id")
        cc_edge_ids=list(cc_edge_ids.values())
        for tank in tankcol:
            if tank in cc:
                edge_sensor_dict.update({edge_id:sensor_switch[tank] for edge_id in cc_edge_ids})

    # manually add removed edges, forward and backward
    cc_dict={
        'P375':sensor_switch["T3"],
        'P399':sensor_switch["T6"],
        'P397':sensor_switch["T5"],
        'P467':sensor_switch["T2"]
    }
    cc_dict.update({
        'P375_rev':sensor_switch["T3"],
        'P399_rev':sensor_switch["T6"],
        'P397_rev':sensor_switch["T5"],
        'P467_rev':sensor_switch["T2"]
    })
    edge_sensor_dict.update(cc_dict)

    # write the dictionary
    with open(path.join(data_path,"edge_sensor_xwalk.csv"), 'w', newline='') as csvfile:
        fieldnames=['edge_id','sensor_set_id']
        writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for edge_id,sensor_id in edge_sensor_dict.items():
            writer.writerow({'edge_id':edge_id,'sensor_set_id':sensor_id})

def get_epath(npath):
    plen=len(npath)
    return [(npath[i],npath[i+1]) for i in range(plen-1)]

def sum_neg(series):
    return reduce(lambda a,b: a+b if (b<0) else a, series)

######################## PIPELINE

def network_gen_unif(intnet,alpha,filepath):
    """
    ### Keywords
    *`intnet` - interdiction network
    *`alpha` - percentage of demand to deactivate
    """
    dem=intnet.get_demand()
    active=[j for (j,t),demand in dem.items() if demand>0]
    sample_size=int(np.ceil(alpha*len(active)))
    null_nodes=list(np.random.choice(active,sample_size,replace=False))
    intnet.lights_off(null_nodes)
    intnet.to_csv(filepath)
    intnet.lights_on()

def network_gen_bfs(intnet,seed_node,edge_depth,filepath):
    """
    ### Keywords
    *`intnet` - interdiction network
    *`seed_node` - initial node of BFS
    *`edge_depth` - depth of BFS
    """
    assert seed_node in intnet.G.nodes
    dem=intnet.get_demand()
    _,_,junctions=intnet.original_wn.get_nodes()
    active=[j for (j,t),demand in dem.items() if demand>0]

    # build  bfs tree and grab nodes to desired depth
    bt=bfs_tree(intnet.G,seed_node)
    keep,level=[seed_node],[seed_node]
    for cur_depth in range(edge_depth):
        # sweep next tier of BFS tree
        level=[n for node in level for n in bt.neighbors(node)]
        keep=np.hstack((keep,level))
    keep=[n for n in list(keep) if n in junctions]

    # to target these nodes, turn off everything else
    null_nodes=[n for n in active if n not in keep]
    intnet.lights_off(null_nodes)
    intnet.to_csv(filepath)
    intnet.lights_on()
    return keep, null_nodes

def trial_loop(intnet_inputfile, output_dir, filename_root, num_networks, unif_args, bfs_args):
    """
    Generate a bunch of networks to interdict, write to disc.
    ### Keywords
    *`intnet_inputfile` - read in wntr
    *`output_dir` - write networks here
    *`filename_root` - beginning of each network filename
    *`num_networks` - number of demands to simulate
    *`unif_args` - num_trials, percentages
    *`bfs_args` - num_trials, edge_depths
    ### Writes
    *full network for each network trial (CSV)
    *lights-off network for each unif and bfs trial (CSV)
    *trials summary (CSV)
    """
    # initialize
    unif_trials,percentages=unif_args
    bfs_trials,edge_depths=bfs_args
    intnet=InterdictionNetwork(intnet_inputfile)

    # set up directory - don't overwrite old results!
    try:
        os.makedirs(output_dir)
    except OSError as e:
        raise e
    rootpath=path.join(output_dir,filename_root)

    # build trials and record trial info
    trial=0
    trial_type,trial_param=[],[]
    for _ in range(num_networks):
        intnet.sim()
        trial+=1
        intnet.to_csv("%s_%d.csv"%(rootpath,trial))

        # full trial info
        trial_type.append("full")
        trial_param.append("")

        # unif trials
        for _ in range(unif_trials):
            for alpha in percentages:
                trial+=1
                filepath="%s_%d.csv"%(rootpath,trial)
                network_gen_unif(intnet,alpha,filepath)
                trial_type.append("unif")
                trial_param.append(alpha)

        # bfs trials
        for edge_depth in edge_depths:
            # generate all the seeds first to ensure no duplicates
            dem=intnet.get_demand()
            active=[j for (j,t),demand in dem.items() if demand>0]
            seed_nodes=np.random.choice(active,bfs_trials,replace=False)
            for seed_node in seed_nodes:
                trial+=1
                filepath="%s_%d.csv"%(rootpath,trial)
                network_gen_bfs(intnet,seed_node,edge_depth,filepath)
                trial_type.append("bfs")
                trial_param.append(edge_depth)

    # record trial info
    trials=[i for i in range(1,trial+1)]
    trial_dat=pd.DataFrame(np.vstack((trials,trial_type,trial_param)),columns=["trial_id","trial_type","trial_param"])
    trial_dat.to_csv(path.join(output_dir,"trial_info.csv"))


def main():
    FILEPATH=path.join(get_data_path(),"CTOWN.INP")
    wn=WaterNetwork(FILEPATH)
    res=wn.sim_demand()
    dem=res.node['demand']
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

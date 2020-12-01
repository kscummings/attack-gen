"""
Build and solve interdiction model
Translate interdiction results into attack strategies
"""
using CSV, DataFrames

# function to read in the interdiction network
# we need: topology, sink, source, source edges, sink edges, fortified edges
struct InterdictionNetwork
    # full topology
    V::Vector{Symbol}                           # nodes
    E::Vector{Symbol}                           # edge ids
    OD::Dict{Symbol,Tuple{Symbol,Symbol}}       # edge ID => OD pair

    # auxiliary topology
    source::Dict{Symbol,Bool}                   # edge => 1(source edge)
    sink::Vector{Symbol}                        # edge => 1(sink edge)
    fortified::Vector{Symbol}                   # edge => 1(not interdictable)

    # network properties
    cap::Dict{Symbol,Real}                      # edge id => capacity
end

"""
IN constructor
### Keywords
*`filepath` - location of network csv
"""
function InterdictionNetwork(
        filepath
    )
    dat=CSV.read(filepath,DataFrame)

    dat.edge_id=Symbol.(dat.edge_id)
    dat.origin=Symbol.(dat.origin)
    dat.dest=Symbol.(dat.dest)

    # compile sets
    N=nrow(dat)
    E=dat[!,:edge_id]
    V=unique(vcat(dat[!,:origin],dat[!,:dest]))

    # compile properties
    OD=Dict{Symbol,Tuple{Symbol,Symbol}}(dat[i,:edge_id]=>(dat[i,:origin],dat[i,:dest]) for i in 1:N)
    source=Dict{Symbol,Bool}(dat[i,:edge_id]=>dat[i,:source] for i in 1:N)
    sink=Dict{Symbol,Bool}(dat[i,:edge_id]=>dat[i,:sink] for i in 1:N)
    fortified=Dict{Symbol,Bool}(dat[i,:edge_id]=>dat[i,:fortified] for i in 1:N)

    # build capacities

end

# function to compute shortest paths from reservoir to each tank, to disqualify key edges from interdiction

# function to build interdiction model

# function to solve interdiction model

# function to translate interdiction solution to sensor attack strategy and write to disc (read in crosswalk between edges and sensor sets)

using CSV, DataFrames, Gurobi, JuMP
"""
Build and solve interdiction model
Translate interdiction results into attack strategies
"""

function get_data_path()
    filepath=joinpath(dirname(dirname(dirname(@__DIR__))),"data-path.txt")
    # Read first line.
    if isfile(filepath)
        open(filepath) do f
            return strip(readline(f))
        end
    else
        @warn "Please write path to data directory in a txt file called `data-path.txt` and store in the transit-alliance root directory."
    end
end

GUROBI_ENV=Gurobi.Env()
EDGE_XWALK_FILEPATH=joinpath(get_data_path(),"edge_sensor_xwalk.csv")

struct InterdictionNetwork
    # full topology
    V::Vector{Symbol}                           # nodes
    E::Vector{Symbol}                           # edge ids
    OD::Dict{Symbol,Tuple{Symbol,Symbol}}       # edge ID => OD pair

    # auxiliary topology
    source::Dict{Symbol,Bool}                   # edge => 1(source edge)
    sink::Dict{Symbol,Bool}                     # edge => 1(sink edge)
    fortified::Dict{Symbol,Bool}                # edge => 1(not interdictable)

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
    cap=Dict{Symbol,Real}(dat[i,:edge_id]=>dat[i,:capacity] for i in 1:N)

    # add demands to cap
    for i in 1:N
        dem=dat[i,:dem]
        if !ismissing(dem)
            cap[dat[i,:edge_id]]=dem
        end
    end

    InterdictionNetwork(V,E,OD,source,sink,fortified,cap)
end

"""
### Keywords
*`intnet` -interdiction network
*`E_hat` - edges eligible for interdiction
*`budget` - number of edges that can be interdicted
"""
function interdiction_model(
        intnet::InterdictionNetwork,
        E_hat::Vector{Symbol},
        budget::Int64,
        edge_pair::Dict
    )
    @assert issubset(E_hat,intnet.E)

    model=JuMP.Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag"=>0))

    JuMP.@variable(model,x[e in E_hat],Bin)
    JuMP.@variable(model,beta[e in E_hat],Bin)
    JuMP.@variable(model,theta[e in intnet.E],Bin)
    JuMP.@variable(model,gamma[k in intnet.V],Bin)

    JuMP.@constraint(model,[e in intnet.E], gamma[intnet.OD[e][1]]-gamma[intnet.OD[e][2]]+theta[e]>=0)
    JuMP.@constraint(model, gamma[:sink]-gamma[:source]>=1)
    JuMP.@constraint(model, [e in E_hat], beta[e]>=theta[e]-x[e])

    JuMP.@objective(model,Min,sum(intnet.cap[e]*theta[e] for e in setdiff(intnet.E,E_hat))+sum(intnet.cap[e]*beta[e] for e in E_hat))

    # interdiction (removal of one removes other direction)
    JuMP.@constraint(model,sum(x[e] for e in E_hat)<=2*budget)
    JuMP.@constraint(model,[e for e in keys(edge_pair)], x[e]==x[edge_pair[e]])

    model
end

"""
obtain an interdiction decision

### Keywords
*`intnet` -interdiction network
*`budget` - number of edges that can be interdicted
*`fortify` - whether to disallow interdiction of fortified edges
*`edge_xwalk` - dictionary ((u,v)=>sensor ID)
"""
function interdiction_decision(
        intnet::InterdictionNetwork,
        budget::Int64,
        fortify::Bool,
        edge_xwalk::Dict
    )
    # build E_hat
    E_offlimits = [edge_id for edge_id in intnet.E if startswith(String(edge_id),"edge")]
    E_fortified = fortify ? [edge_id for edge_id in intnet.E if intnet.fortified[edge_id]] : []
    E_offlimits=vcat(E_offlimits,E_fortified)
    E_hat=setdiff(intnet.E,E_offlimits)

    # solve, obtain interdiction
    model=interdiction_model(intnet,E_hat,budget)
    optimize!(model)
    x=Dict(e=>JuMP.value(model.obj_dict[:x]) for e in E_hat)
end


exw=CSV.read(EDGE_XWALK_FILEPATH, DataFrame)
exw=Dict((exw[i,:origin],exw[i,:dest])=>exw[i,:sensor_set_id] for i in 1:nrow(exw))

# create forward-backward edge pairs
phys_edges=[edge_id for edge_id in intnet.E if !startswith(String(edge_id),"edge") & !endswith(String(edge_id),"_rev")]
edge_pair=Dict(edge_id=>Symbol(join([edge_id,"rev"],"_")) for edge_id in phys_edges)

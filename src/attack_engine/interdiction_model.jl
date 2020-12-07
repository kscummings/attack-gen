using CSV, DataFrames, Gurobi, JuMP, Glob
"""
Build and solve interdiction model
Translate interdiction results into attack strategies
"""

############ INPUTS

BUDGETS=[i for i in 1:2]#1:5
INPUT_DIR="test_13756"

############ MODEL

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

DATA_PATH="/Users/kaylacummings/Dropbox (MIT)/batadal"#get_data_path()
GUROBI_ENV=Gurobi.Env()
EDGE_XWALK_FILEPATH=joinpath(DATA_PATH,"edge_sensor_xwalk.csv")
INPUT_PATH=joinpath(DATA_PATH,INPUT_DIR)

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

    model=JuMP.Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>0))

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
    JuMP.@constraint(model,[e in intersect(keys(edge_pair),E_hat)], x[e]==x[edge_pair[e]])

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
        edge_xwalk::Dict,
        edge_pair::Dict
    )
    # build E_hat
    E_offlimits = [edge_id for edge_id in intnet.E if startswith(String(edge_id),"edge")]
    E_fortified = fortify ? [edge_id for edge_id in intnet.E if intnet.fortified[edge_id]] : []
    E_offlimits=vcat(E_offlimits,E_fortified)
    E_hat=setdiff(intnet.E,E_offlimits)

    # solve, obtain interdiction
    model=interdiction_model(intnet,E_hat,budget,edge_pair)
    optimize!(model)
    x=Dict(e=>JuMP.value(model.obj_dict[:x][e]) for e in E_hat)
    interdicted=[key for (key,value) in x if (value==1) & !endswith(String(key),"rev")]

    # which sensors to mess with?
    sensors=[edge_xwalk[e] for e in interdicted]
    return sensors
end

"""
Read in all of the networks written to a directory
Interdict them
write the results to the same directory
### Keywords
*`trials_dir` - directory containing networks
*`budgets` - list of interdiction budgets to test
*`edge_xwalk` - map between edges and sensor sets
*`edge_pair` - edge ids of directed edges that are paired in undirected graph
"""
function all_trials(
        trials_dir::String,
        budgets::Vector{Int64},
        edge_xwalk::Dict,
        edge_pair::Dict
    )
    # one of these is trial_info
    trial_info_filename=Glob.glob("trial_info.csv",trials_dir)
    filenames=Glob.glob("*.csv",trials_dir)
    filter!(f -> f!=trial_info_filename[1],filenames)

    trials,b,fortify=[],[],[]
    s1,s2,s3,s4,s5=[],[],[],[],[] # there's definitely a better way to do this
    for f in filenames
        intnet=InterdictionNetwork(f)
        trial_num=match(r"(\d+)", replace(f,trials_dir=>""))
        trial_num=parse(Int,trial_num.match)
        for budget in budgets
            for fort in [true,false]
                s=interdiction_decision(intnet,budget,fort,edge_xwalk,edge_pair)
                push!(b,budget)
                push!(fortify,fort)
                push!(trials,trial_num)

                # wow you are so good at coding kayla
                if !isempty(s)
                    push!(s1,sum(i==1 for i in s))
                    push!(s2,sum(i==2 for i in s))
                    push!(s3,sum(i==3 for i in s))
                    push!(s4,sum(i==4 for i in s))
                    push!(s5,sum(i==5 for i in s))
                else
                    push!(s1,false)
                    push!(s2,false)
                    push!(s3,false)
                    push!(s4,false)
                    push!(s5,false)
                end
            end
        end
    end

    # build dataframe
    res=DataFrames.DataFrame(trial_id=trials,budget=b,fortify=fortify,s1=s1,s2=s2,s3=s3,s4=s4,s5=s5)
    CSV.write(joinpath(trials_dir,"results.csv"),res)
end


############ PIPELINE

function main()
    exw=CSV.read(EDGE_XWALK_FILEPATH, DataFrame)
    exw=Dict(Symbol(exw[i,:edge_id])=>exw[i,:sensor_set_id] for i in 1:nrow(exw))

    # create forward-backward edge pairs
    phys_edges=[edge_id for edge_id in intnet.E if !startswith(String(edge_id),"edge") & !endswith(String(edge_id),"_rev")]
    edge_pair=Dict(edge_id=>Symbol(join([edge_id,"rev"],"_")) for edge_id in phys_edges)

    all_trials(INPUT_PATH,BUDGETS,exw,edge_pair)
end


main()

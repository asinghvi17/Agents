module Agents

using Flux, Distributions, PyCall, DataStructures
using Flux: @forward, data
using Flux.Tracker: TrackedReal

Memory{State} = Tuple{State, Int, Float32, State, Bool}

abstract type AbstractAgent{State} end
abstract type AbstractEnvironment{State} end

include("layers.jl")
include("gym.jl")
include("actor_critic.jl")
include("simulation.jl")

end

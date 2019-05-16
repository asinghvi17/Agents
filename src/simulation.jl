struct Simulation{State, A<:AbstractAgent{State}, E<:AbstractEnvironment{State}}
    agent::A
    env::E
end

Simulation(agent::AbstractAgent{S}, env::AbstractEnvironment{S}) where S =
    Simulation{S, typeof(agent), typeof(env)}(agent, env)

Base.IteratorSize(::Type{Simulation{S, A, E}}) where {S, A, E} =
    Base.IsInfinite()

Base.eltype(::Type{Simulation{S, A, E}}) where {S, A, E} = Memory{S}

function Base.iterate(simulation::Simulation, state=reset(simulation.env))
    action = select_action!(simulation.agent, state)
    next_state, reward, done, _ = step(simulation.env, action)
    new_state = done ? reset(simulation.env) : next_state
    memory = (state, action, reward, next_state, done)
    remember!(simulation.agent, memory)
    memory, new_state
end


struct Episodes{Simulation}
    n::Int
    simulation::Simulation
end

Base.IteratorSize(::Type{Episodes{S}}) where S = Base.SizeUnknown()

Base.eltype(::Type{Episodes{S}}) where {S} = eltype(S)

function Base.iterate(episodes::Episodes)
    result = iterate(episodes.simulation)
    isnothing(result) && return nothing
    memory, simulation_state = result
    memory, (0, simulation_state)
end

function Base.iterate(episodes::Episodes, state)
    episode, simulation_state = state
    episode >= episodes.n && return nothing
    result = iterate(episodes.simulation, simulation_state)
    isnothing(result) && return nothing
    memory, simulation_state = result
    memory, (memory[5] ? episode + 1 : episode, simulation_state)
end



struct Frames{Simulation}
    n::Int
    simulation::Simulation
end

Base.IteratorSize(::Type{Frames{S}}) where S = Base.SizeUnknown()

Base.eltype(::Type{Frames{S}}) where {S} = eltype(S)

function Base.iterate(frames::Frames)
    result = iterate(frames.simulation)
    isnothing(result) && return nothing
    memory, simulation_state = result
    memory, (1, simulation_state)
end

function Base.iterate(frames::Frames, state)
    frame, simulation_state = state
    frame >= frames.n && return nothing
    result = iterate(frames.simulation, simulation_state)
    isnothing(result) && return nothing
    memory, simulation_state = result
    memory, (frame + 1, simulation_state)
end

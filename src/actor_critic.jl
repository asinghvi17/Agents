struct ActorCritic{State, Model, Optimizer} <: AbstractAgent{State}
    model::Model
    optimizer::Optimizer
    memory::CircularBuffer{Memory{State}}
    minibatch_size::Int
    discount_factor::Float32
end

function ActorCritic(state_space::Integer, action_space::Integer;
                     hidden_units=20, memory_capacity=1_000, minibatch_size=32,
                     discount_factor=0.9f0)
    model = Chain(Dense(state_space, hidden_units, relu),
                  FanOut(2),
                  Parallel(Chain(Dense(hidden_units, action_space), softmax),
                           Dense(hidden_units, 1)))
    optimizer = ADAM()

    State, Model, Optimizer = Vector{Float32}, typeof(model), typeof(optimizer)

    memory = CircularBuffer{Memory{State}}(memory_capacity)
    ActorCritic{State, Model, Optimizer}(
        model, optimizer, memory, minibatch_size, discount_factor)
end

function select_action!(agent::ActorCritic{State}, state::State) where State
    action_probabilities, _ = agent.model(state)
    rand(Categorical(action_probabilities))
end

function improve!(agent::ActorCritic)
    minibatch = sample(agent.memory, agent.minibatch_size, replace=false)
    log_probabilities = TrackedReal{Float32}[]
    advantages = TrackedReal{Float32}[]
    for (state, action, reward, next_state, done) ∈ minibatch
        _, next_state_value = agent.model(next_state)
        action_probabilities, state_value = agent.model(state)
        distribution = Categorical(action_probabilities)
        log_probability = loglikelihood(distribution, [action])
        advantage = reward + agent.discount_factor * data(next_state_value)[1] - state_value[1]
        push!(log_probabilities, log_probability)
        push!(advantages, advantage)
    end
    log_probabilities, advantages
end

function remember!(agent::ActorCritic, memory::Memory)
    push!(agent.memory, memory)
    length(agent.memory) >= agent.minibatch_size && improve!(agent)
end

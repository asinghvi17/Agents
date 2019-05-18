struct ActorCritic{Network, Optimizer}
    network::Network
    optimizer::Optimizer
    log_probabilities::Vector{TrackedReal{Float32}}
    values::Vector{TrackedReal{Float32}}
    rewards::Vector{Float32}
    γ::Float32
end

function ActorCritic(state_space::Integer, action_space::Integer;
                     hidden=40, η=0.001, γ=0.9)
    network = Chain(LSTM(state_space, hidden),
                    LSTM(hidden, hidden),
                    Dense(hidden, hidden, elu),
                    FanOut(2),
                    Parallel(Chain(Dense(hidden, action_space), softmax),
                             Dense(hidden, 1)))
    optimizer = ADAM(η)
    log_probabilities = TrackedReal{Float32}[]
    values = TrackedReal{Float32}[]
    rewards = Float32[]
    ActorCritic(network, optimizer, log_probabilities,
                values, rewards, Float32(γ))
end

function select_action!(agent::ActorCritic, state)
    probabilities, value = agent.network(state)
    distribution = Categorical(probabilities)
    action = rand(distribution)
    push!(agent.log_probabilities, loglikelihood(distribution, [action]))
    push!(agent.values, value[1])
    action
end

function discount(rewards::AbstractVector{T}, γ::T) where T
    discounted = similar(rewards)
    running_sum = T(0.0)
    for i in length(rewards):-1:1
        running_sum = running_sum * γ + rewards[i]
        discounted[i] = running_sum
    end
    discounted
end

normalize(xs::AbstractVector{T}) where T =
    (xs .- mean(xs)) / (std(xs) + eps(T))

function improve!(agent::ActorCritic)
    returns = normalize(discount(agent.rewards, agent.γ))
    advantage = returns - agent.values
    π_loss = mean(-agent.log_probabilities .* advantage)
    V_loss = mean(advantage.^2)
    Flux.reset!(agent.network)
    θ = params(agent.network)
    Δ = gradient(() -> π_loss + V_loss, θ)
    update!(agent.optimizer, θ, Δ)
    empty!(agent.log_probabilities)
    empty!(agent.values)
    empty!(agent.rewards)
end

function remember!(agent::ActorCritic, (_, _, reward, _, done))
    push!(agent.rewards, reward)
    done && improve!(agent)
end

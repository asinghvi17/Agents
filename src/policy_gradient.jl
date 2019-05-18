struct PolicyGradient{Policy, Optimizer}
    π::Policy
    optimizer::Optimizer
    log_probabilities::Vector{TrackedReal{Float32}}
    rewards::Vector{Float32}
    γ::Float32
end

function PolicyGradient(state_space::Integer, action_space::Integer;
                        hidden=40, η=0.001, γ=0.9)
    π = Chain(LSTM(state_space, hidden),
              Dense(hidden, hidden, elu),
              Dense(hidden, hidden, elu),
              Dense(hidden, action_space), softmax)
    optimizer = ADAM(η)
    log_probabilities = TrackedReal{Float32}[]
    rewards = Float32[]
    PolicyGradient(π, optimizer, log_probabilities, rewards, Float32(γ))
end

function select_action!(agent::PolicyGradient, state)
    distribution = Categorical(agent.π(state))
    action = rand(distribution)
    push!(agent.log_probabilities, loglikelihood(distribution, [action]))
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

function improve!(agent::PolicyGradient)
    returns = normalize(discount(agent.rewards, agent.γ))
    π_loss = mean(-agent.log_probabilities .* returns)
    Flux.reset!(agent.π)
    θ = params(agent.π)
    Δ = gradient(() -> π_loss, θ)
    update!(agent.optimizer, θ, Δ)
    empty!(agent.log_probabilities)
    empty!(agent.rewards)
end

function remember!(agent::PolicyGradient, (_, _, reward, _, done))
    push!(agent.rewards, reward)
    done && improve!(agent)
end

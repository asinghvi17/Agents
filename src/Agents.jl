module Agents

using Flux, PyCall, Distributions, Makie
using Flux.Tracker: TrackedReal, gradient, update!
using Flux: @forward, params
using Juno: @progress

AbstractPlotting.inline!(true)

include("layers.jl")
include("gym.jl")
include("policy_gradient.jl")


function simulate!(agent, env; episodes=1, render=false, graph=true)
    rewards = Float32[]
    @progress "simulate!" for _ ∈ 1:episodes
        episode_reward = 0.0f0
        state = env.reset()
        done = false
        while !done
            action = select_action!(agent, state)
            next_state, reward, done, _ = env.step(action - 1)
            remember!(agent, (state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            render && env.render()
        end
        push!(rewards, episode_reward)
    end
    graph ? plot(rewards) : mean(rewards)
end


end

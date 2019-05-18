module Agents

using Flux, PyCall, Distributions, Makie
using Flux.Tracker: TrackedReal, gradient, update!
using Flux: @forward, params
using Juno: @progress
using Makie: update!

include("layers.jl")
include("gym.jl")
include("policy_gradient.jl")
include("actor_critic.jl")

function simulate!(agent, env; episodes=1, render=false, graph=true)
    rewards = Node([Point2f0(0)])
    graph && (scene = lines(rewards, linewidth=10); display(scene))
    @progress "simulate!" for episode ∈ 1:episodes
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
        push!(rewards[], Point2f0(episode, episode_reward))
        graph && (Makie.update!(scene); Makie.update_limits!(scene))
    end
end

end

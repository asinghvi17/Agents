include("../src/Agents.jl")

env = Agents.gym.make("CartPole-v0")

agent = Agents.ActorCritic(4, 2)

agent = Agents.PolicyGradient(4, 2)

Agents.simulate!(agent, env, episodes=100)

Agents.simulate!(agent, env, episodes=5, graph=false, render=true)

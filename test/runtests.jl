include("../src/Agents.jl")

env = Agents.gym.make("CartPole-v0")

agent = Agents.PolicyGradient(4, 2)

Agents.simulate!(agent, env, episodes=200)

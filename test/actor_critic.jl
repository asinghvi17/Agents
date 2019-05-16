using Lazy

include("../src/Agents.jl")

env = Agents.GymEnvironment("CartPole-v0")

agent = Agents.ActorCritic(4, 2)

@>> begin
    Agents.Simulation(agent, env)
    Agents.Frames(32)
    foreach(_ -> nothing)
end


agent

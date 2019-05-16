const gym = PyNULL()

struct GymEnvironment{State} <: AbstractEnvironment{State}
    env::PyObject
end

function GymEnvironment(name::AbstractString)
    env = gym.make(name)
    GymEnvironment{typeof(Float32.(env.reset()))}(env)
end

Base.reset(gym::GymEnvironment) = Float32.(gym.env.reset())

function Base.step(gym::GymEnvironment, action::Integer)
    state, reward, done, info = gym.env.step(action - 1)
    Float32.(state), Float32(reward), done, info
end

render(gym::GymEnvironment) = gym.env.render()

Base.close(gym::GymEnvironment) = gym.env.close()

__init__() = copy!(gym, pyimport("gym"))

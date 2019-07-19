include("neuron.jl")

mutable struct LIF <: Neuron
    u::Float64
    rest_u::Float64
    reset_u::Float64
    tau::Float64
    thresh::Float64
    resist::Float64
end


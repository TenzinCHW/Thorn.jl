include("neuron.jl")

mutable struct LIF{T<:AbstractFloat} <: ProcessingNeuron
    u::T
    rest_u::T
    reset_u::T
    tau::T
    thresh::T
    resist::T
end


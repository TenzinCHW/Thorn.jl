include("neuron.jl")

struct RateInputNeuron{T<:AbstractFloat} <: InputNeuron
end

function generate_static_input_rate(neuron::RateInputNeuron{T}, sensor_inp::T) where T
end

struct PoissonInputNeuron{T<:AbstractFloat} <: InputNeuron
end

function generate_static_input_poisson(neuron::PoissonInputNeuron{T}, sensor_inp::T) where T
end


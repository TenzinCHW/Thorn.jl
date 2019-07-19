include("../spike.jl")
include("../neurons/neuron.jl")

struct NeuronPopulation{S<:AbstractFloat, T<:AbstractFloat}
    neurons::Array{Neuron, 1}
    potential_function::Function # This is the general solution to the membrane potential DE
    weights::Array{S, 2}
    weight_update::Function
    out_spikes::Array{Spike, 1}
    delay_::T
end


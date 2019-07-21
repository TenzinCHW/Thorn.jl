include("../spike.jl")
include("../neurons/neuron.jl")

abstract type NeuronPopulation end

mutable struct ProcessingNeuronPopulation{S<:AbstractFloat, T<:AbstractFloat} <: NeuronPopulation
    neurons::Array{ProcessingNeuron, 1}
    potential_function::Function # This is the general solution to the membrane potential DE
    intrapop_weights::Array{S, 2}
    weight_update::Function
    out_spikes::Array{Spike, 1}
    last_spike::Spike
end

function process_spike()
    # TODO Given a population, an input spike and the corresponding weights, compute new state of each neuron in pop
    # Assign the spike to the last_spike variable of the pop
end

function output_spikes()
    # TODO Compute a list of spikes output and only insert those that spike before the next spike in out_spikes array of the pop into that same array
    # Update the given weights
end

struct InputNeuronPopulation <: NeuronPopulation
    neurons::Array{InputNeuron, 1}
    generate_static_input::Function # Should take in a raw sensor input and an InputNeuron and gnerate output spikes, putting them into out_spikes
    out_spikes::Array{Spike, 1}
end


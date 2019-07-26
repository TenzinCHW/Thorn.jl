include("../spike.jl")
include("../neurons/neuron.jl")

abstract type NeuronPopulation end

mutable struct ProcessingNeuronPopulation{S<:AbstractFloat, T<:AbstractFloat} <: NeuronPopulation
    id::UInt
    neurons::Array{ProcessingNeuron, 1}
    state_update::Function
    output_spike::Function # Takes in single neuron, produces next spike produced by it or nothing
    weight_update::Function
    out_spikes::Array{Spike, 1}
    last_spike::Spike

    function ProcessingNeuronPopulation(id::UInt, neuron_type::DataType, sz::UInt, state_update::Function, output_spike::Function, weight_update::Function)
        neurons = [neuron_type(i) for i in 1:sz]
        new(id, neurons, state_update, output_spike, weight_update, Array{Spike, 1}, nothing)
    end
end

function process_spike!(pop::NeuronPopulation, weights::Array{AbstractFloat, 1}, spike::Spike)
    pop.state_update.(pop.neurons, weights, spike, pop.last_spike)
    # Assign the spike to the last_spike variable of the pop
    pop.last_spike = spike
end

function output_spike!(pop::NeuronPopulation, next_spike::Spike)
    new_spikes = flatten(pop.output_spike.(pop.neurons, next_spike))
    # filter this based on the timing of next_spike coming into the pop then sort the filtered array
    for s in sort(filter!(x->x!=nothing, new_spikes), rev=true)
        push!(pop.out_spikes, s) # This is faster than concat by 8x
    end
end

function flatten(arr::Array{Any, 1})::Array{Spike, 1}
    out = []
    for item in arr
        if (item <: Array)
            for i in item
                push!(out, i)
            end
        else
            if (item != nothing)
                push!(out, item)
            end
        end
    out
end

function update_weights!(pop::NeuronPopulation, weights::Array{AbstractFloat, 1}, spike::Spike, next_spike::Spike)
    # Update the given weights
    weights = pop.weight_update.(pop.neurons, weights, spike, next_spike)
end

struct InputNeuronPopulation <: NeuronPopulation
    neurons::Array{InputNeuron, 1}
    generate_static_input::Function # Should take in a raw sensor input and an InputNeuron and gnerate output spikes, putting them into out_spikes
    out_spikes::Array{Spike, 1}
end

# TODO write function for generate_state_input


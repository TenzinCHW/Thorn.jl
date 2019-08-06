include("../spike.jl")
include("../neuron.jl")

abstract type NeuronPopulation end

mutable struct ProcessingNeuronPopulation{T<:AbstractFloat} <: NeuronPopulation
    id::UInt
    neurons::Array{ProcessingNeuron, 1}
    state_update::Function
    output_spike::Function # Takes in single neuron, produces next spike produced by it or nothing
    weight_update::Function
    lr::T # Learning rate
    out_spikes::Array{Spike, 1}
    last_spike::Spike

    function ProcessingNeuronPopulation(id::Int, neuron_type::Module, sz::Int, weight_update::Function, lr::AbstractFloat)
        if (id < 1 || sz < 1)
            error("sz and id must be > 0")
        end
        neurons = [neuron_type.NeuronStruct(i) for i in 1:sz]
        new{typeof(lr)}(UInt(id), neurons, neuron_type.state_update, neuron_type.output_spike, weight_update, lr, Array{Spike, 1}, nothing)
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
    for s in sort(filter!(x->x!=nothing, new_spikes), by=x->x.time, rev=true)
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
    end
    out
end

function update_weights!(pop::NeuronPopulation, weights::Array{AbstractFloat, 1}, spike::Spike, next_spike::Spike)
    weights = pop.weight_update.(pop.neurons, pop.lr, weights, spike, next_spike)
end

struct InputNeuronPopulation <: NeuronPopulation
    id::UInt
    neurons::Array{InputNeuron, 1}
    generate_func::Function # Should take in a raw sensor input and an InputNeuron and gnerate output spikes, putting them into out_spikes
    spiketype
    out_spikes::Array{Spike, 1}

    function InputNeuronPopulation(id::Int, neurontype::Module, sz::Int, spiketype)
        if (id < 1 || sz < 1)
            error("id and sz must be > 0")
        end
        if !(neurontype<:InputNeuron)
            error("neuron_type must be a type of InputNeuron")
        end

        neurons = [neurontype.NeuronStruct(i) for i in 1:sz]
        new(UInt(id), neurons, neurontype.generate_input, spiketype, Array{Spike, 1})
    end
end


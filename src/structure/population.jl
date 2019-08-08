abstract type NeuronPopulation end

mutable struct ProcessingNeuronPopulation{T<:AbstractFloat} <: NeuronPopulation
    id::UInt
    neurons::Array{ProcessingNeuron, 1}
    weight_update::Function
    lr::T # Learning rate
    out_spikes::Array{Spike, 1}
    last_spike::Union{Spike, Nothing}

    function ProcessingNeuronPopulation(id::Int, neurontype::UnionAll, sz::Int, weight_update, lr::AbstractFloat)
        (id < 1 || sz < 1) ? error("sz and id must be > 0") : nothing
        !(neurontype<:ProcessingNeuron) ? error("neurontype must be a subtype of ProcessingNeuron") : nothing
        !isa(weight_update, Function) ? error("weight_update must be a function") : nothing
        neurons = [neurontype(i) for i in 1:sz]
        new{typeof(lr)}(UInt(id), neurons, weight_update, lr, Spike[], nothing)
    end
end

function process_spike!(pop::NeuronPopulation, weights::Array{T, 1}, spike::Spike) where T<:AbstractFloat
    for i in eachindex(pop.neurons)
        state_update!(pop.neurons[i], weights[i], spike, pop.last_spike)
    end
    # Assign the spike to the last_spike variable of the pop
    pop.last_spike = spike
end

function output_spike!(pop::NeuronPopulation, spike::T, next_spike::T) where T<:Spike
    new_spikes = flatten([output_spike!(n, spike, next_spike) for n in pop.neurons])
    # filter this based on the timing of next_spike coming into the pop then sort the filtered array
    for s in sort(new_spikes, by=x->x.time, rev=true)
        push!(pop.out_spikes, s) # This is faster than concat by 8x
    end
end

function flatten(arr::Array)
    out = Spike[]
    for item in arr
        if (isa(item, Array))
            for i in item
                push!(out, i)
            end
        else
            item != nothing ? push!(out, item) : nothing
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
    spiketype::UnionAll
    out_spikes::Array{Spike, 1}

    function InputNeuronPopulation(id::Int, neurontype::UnionAll, sz::Int, spiketype::UnionAll)
        (id < 1 || sz < 1) ? error("id and sz must be > 0") : nothing
        !(neurontype<:InputNeuron) ? error("neuron_type must be a type of InputNeuron") : nothing

        neurons = [neurontype(i) for i in 1:sz]
        new(UInt(id), neurons, spiketype, spiketype[])
    end
end

function generate_input_spikes!(input_pop::InputNeuronPopulation, data::Array{T, 1}, maxval::T) where T<:AbstractFloat
    # Given a set of raw sensor data and an InputNeuronPopulation, generate input spikes using the generate_input function for the InputNeuronPopulation
    # Length of array of raw data must match corresponding InputNeuronPopulation
    if !(isequal(length(data), length(input_pop.neurons)))
        error("Dimensions of data must match number of neurons")
    end
    # Call generate_input for each vector of inputs in second dim, broadcast across the first dimension
    spikes = flatten(generate_input.(input_pop.neurons, data, maxval, input_pop.spiketype))
    for s in spikes
        push!(input_pop.out_spikes, s)
    end
    # Must sort the out_spikes of each InputNeuronPopulation after generating the spikes in reverse order based on time property of each spike
    sort!(input_pop.out_spikes, by=x->x.time, rev=true);
end


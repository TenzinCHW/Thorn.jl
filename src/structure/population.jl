abstract type NeuronPopulation end

Base.length(p::NeuronPopulation) = 1

Base.iterate(p::NeuronPopulation) = p, nothing
Base.iterate(p::NeuronPopulation, n::Nothing) = nothing

abstract type ProcessingPopulation <: NeuronPopulation end

function update_weights!(pop::ProcessingPopulation, weights::SubArray{T, 1}, spike::Spike, next_spike::Union{Spike, Nothing}) where T<:AbstractFloat
    weights .= pop.weight_update.(pop, weights, pop.last_out, spike, next_spike)
end

abstract type InputPopulation <: NeuronPopulation end

function generate_input_spikes!(input_pop::InputPopulation, data::Array{T, 1}, maxval::T) where T<:AbstractFloat
    # Given a set of raw sensor data and an InputNeuronPopulation, generate input spikes using the generate_input function for the InputNeuronPopulation
    # Length of array of raw data must match corresponding InputNeuronPopulation
    if !(isequal(length(data), input_pop.length))
        error("Dimensions of data must match number of neurons")
    end
    # Call generate_input for each vector of inputs in second dim, broadcast across the first dimension
    spikes = generate_input.(input_pop, 1:input_pop.length, data, maxval)
    for s in Iterators.flatten(spikes)
        push!(input_pop.out_spikes, s)
    end
    # Must sort the out_spikes of each InputNeuronPopulation after generating the spikes in reverse order based on time property of each spike
    sort!(input_pop.out_spikes, by=x->x.time, rev=true);
end

function insertsorted(out_spikes::Array{S, 1}, spikes::Array{S, 1}) where S<:Spike
    for s in sort(spikes, by=x->x.time, rev=true)
        if isempty(out_spikes) || s.time <= out_spikes[end].time
            push!(out_spikes, s)
        else
            index = searchsortfirst(out_spikes, s, by=x->x.time, rev=true)
            insert!(pop.out_spikes, index, s)
        end
    end
end


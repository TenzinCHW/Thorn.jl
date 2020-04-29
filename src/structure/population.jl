abstract type NeuronPopulation end

Base.length(p::NeuronPopulation) = 1

Base.iterate(p::NeuronPopulation) = p, nothing
Base.iterate(p::NeuronPopulation, n::Nothing) = nothing

abstract type ProcessingPopulation <: NeuronPopulation end

function InputPopulationPair(args...)
    k = args[end]
    k[:sign] = -1 # Only need to modify sign for neg, since default sign is pos
    (args), (args[1:end-1]..., k)
end

function update_weights!(pop::ProcessingPopulation, weights::Weights, spike::Spike) where T<:AbstractFloat
    for s in pop.out_spikes
        #if s.time < spike.time
            pop.weight_update(weights.value, pop, spike, s)
        #end
    end
end

function update_weights!(srcpop::NeuronPopulation, dstpop::ProcessingPopulation, weights::Weights, newspike::Spike) where T<:AbstractFloat
    #TODO update weights for each spike in populations that dst depends on for each new spike produced by dst
    for s in srcpop.out_spikes.items
        if s.time < newspike.time
            dstpop.weight_update(weights.value, dstpop, s, newspike)
        end
    end
end

abstract type InputPopulation <: NeuronPopulation end

function generate_input_spikes!(input_pop::InputPopulation, data::Array{T, 2}, maxval::T) where T<:AbstractFloat
    # Given a set of raw sensor data and an InputNeuronPopulation, generate input spikes using the generate_input function for the InputNeuronPopulation
    # First dim of raw data must match length of corresponding InputNeuronPopulation
    if !(isequal(size(data, 1), input_pop.length))
        error("Dimensions of data must match number of neurons")
    end
    # Call generate_input for each vector of inputs in second dim, broadcast across the first dimension
    spikes = [generate_input(input_pop, i, data[i, :], maxval) for i in 1:input_pop.length]
    for s in Iterators.flatten(spikes)
        push!(input_pop.out_spikes, s)
    end
    # Must sort the out_spikes of each InputNeuronPopulation after generating the spikes in reverse order based on time property of each spike
    sort!(input_pop.out_spikes.items, by=x->x.time);
end


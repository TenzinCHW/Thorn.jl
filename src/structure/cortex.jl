include ("population.jl")

struct Cortex{T<:AbstractFloat}
    input_populations::Array{InputNeuronPopulation, 1}
    populations::Array{NeuronPopulation, 1}
    weights::Array{Union{Nothing, Array{T, 2}}, 2} # Input is along 2nd dimension
    connectivity_matrix::BitArray{2}

    # Constructor with connectivity matrix and weight initialisation function
    function Cortex(input_neuron_types::Array{Pair{DataType, UInt}, 1}, neuron_types::Array{Pair{DataType, UInt}, 1}, conn::Array{Bool, 2}, wt_init::Function)
        # Check that neurons are correct type
        if !(all(x -> <:(x, InputNeuron), first.(input_neuron_types)))
            error("input_neuron_types should all contain types of InputNeuron")
        end
        if !(all(x -> <:(x, ProcessingNeuron), first.(neuron_types)))
            error("neuron_types should all contain types of ProcessingNeuron")
        end
        if !(isequal(size(conn)...))
            error("conn must be a square matrix")
        end
        if !(isequal(first(size(conn)), length(input_neuron_types) + length(neuron_types)))
            error("length of one side of conn must be equal to sum of lengths of input_neuron_types and neuron_types")
        end
        new()
    end

    # Constructor with connectivity matrix (use rand() for generating weights
    function Cortex(input_neuron_types::Array{Pair{DataType, UInt}, 1}, neuron_types::Array{Pair{DataType, UInt}, 1}, conn::Array{Bool, 2})
        Cortex(input_neuron_types, neuron_types, conn, rand)
    end
end

function generate_input_spikes()
    # TODO Given a set of raw sensor data and a Cortex, generate input spikes using the generate_static_input function of each corresponding InputNeuronPopulation
    # Length of array of raw data must match corresponding InputNeuronPopulation
    # If raw sensor data is 2D, the first dimension must represent space and second is time
    # Call generate_static_input for each vector of inputs in second dim, broadcast across the first dimension
    # Must sort the out_spikes property of each InputNeuronPopulation after generating the spikes
end

function process_sample(cortex::Cortex)
    # While any population in a Cortex has spikes in its out_spikes property, call process_next_spike
    while (any(length.(cortex.populations.out_spikes) .> 0))
        process_next_spike(cortex)
    end
end

function process_next_spike(cortex::Cortex)
    # Take the head spike from out_spikes queue of each population with smallest time property
    pop_id, spike = pop_next_spike(cortex.populations)
    inds = findall(cortex.connectivity_matrix[:,pop_id])
    for i in inds
        # Route this spike to the correct populations with their weights using the process_spike function for every population that the spike is routed to
        pop = cortex.populations[i]
        weights = cortex.weights[i,pop_id][:, spike.neuron_index]
        process_spike!(pop, weights, spike)
        # Find next_spike for each of the populations that just processed spikes and call output_spikes! to generate output spikes for each of those populations
        is = findall(cortex.connectivity_matrix[pop.id,:])
        _, next_spike, _ = get_next_spike(cortex.populations[is])
        output_spikes!(pop, next_spike)
        #  Call update_weights! to update the weights for those populations that just processed the spike
        update_weights!(pop, weights, spike, next_spike)
    end
end

function pop_next_spike(pops::Array{NeuronPopulation, 1})
    non_empty_pops, _, ind = get_next_spike(pop)
    non_empty_pops[ind].id, pop!(non_empty_pops[ind].out_spikes)
end

function get_next_spike(pops::Array{NeuronPopulation, 1})
    non_empty_pops = filter(has_out_spikes, cortex.populations)
    lens = num_out_spikes.(non_empty_pops)
    earliest_spikes = getindex.(non_empty_pops, lens)
    s, ind = findmin(earliest_spikes)
    non_empty_pops, s, ind
end

has_out_spikes(pop::NeuronPopulation) = length(pop.out_spikes) > 0

num_out_spikes(pop::NeuronPopulation) = length(pop.out_spikes)


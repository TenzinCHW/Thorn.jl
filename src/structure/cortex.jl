include ("population.jl")

struct Cortex{T<:AbstractFloat}
    input_populations::Array{InputNeuronPopulation, 1}
    populations::Array{NeuronPopulation, 1}
    weights::Dict{Pair{Int, Int}, Array{T, 2}} # Input is along 2nd dimension
    connectivity_matrix::BitArray{2}

    # Constructor with connectivity_matrix and weight initialisation function
    function Cortex(input_neuron_types::Array{Tuple{Module, Int, Function}, 1}, neuron_types::Array{Tuple{Module, Int, Function}, 1}, connectivity_matrix::BitArray{2}, wt_init::Function)
        if !(isequal(size(connectivity_matrix)...))
            error("conn must be a square matrix")
        end
        if !(isequal(first(size(connectivity_matrix)), length(input_neuron_types) + length(neuron_types)))
            error("length of one side of conn must be equal to sum of lengths of input_neuron_types and neuron_types")
        end
        # Make the populations in the order given
        input_populations = Array{InputNeuronPopulation, 1}
        for i in eachindex(input_neuron_types)
            push!(input_populations, InputNeuronPopulation(i, input_neuron_types[i]...))
        end

        populations = Array{NeuronPopulation, 1}
        for i in eachindex(neuron_types)
            push!(populations, ProcessingNeuronPopulation(i, neuron_types[i]...))
        end
        populations = cat(input_populations, populations)

        # Make the weights and insert into a Dict
        weights = Dict(Pair{Int, Int}, Array{AbstractFloat, 2})
        for j in 1:size(connectivity_matrix, 2)
            for i in 1:size(connectivity_matrix, 1)
                if (connectivity_matrix[i, j])
                    weights[j=>i] = wt_init(length(populations[i]), length(populations[j]))
                end
            end
        end
        new(input_populations, populations, weights, connectivity_matrix)
    end

    # Constructor with connectivity_matrix (use rand() for generating weights
    function Cortex(input_neuron_types::Array{Tuple{Module, Int, Function}, 1}, neuron_types::Array{Tuple{Module, Int, Function}, 1}, conn::BitArray{2})
        Cortex(input_neuron_types, neuron_types, conn, rand)
    end
end

function process_sample!(cortex::Cortex, input::Array{Array{Number, 1}, 1})
    # Generate spike data from input; each array is for each corresponding input pop
    for i in eachindex(input)
        generate_input_spikes!(cortex.input_populations[i], input[i])
    end
    # While any population in a Cortex has spikes in its out_spikes property, call process_next_spike
    while (any(length.(cortex.populations.out_spikes) .> 0))
        process_next_spike!(cortex)
    end
end

function generate_input_spikes!(input_pop::InputNeuronPopulation, data::Array{Number, 1})
    # Given a set of raw sensor data and an InputNeuronPopulation, generate input spikes using the generate_func function of the InputNeuronPopulation
    # Length of array of raw data must match corresponding InputNeuronPopulation
    if !(isequal(length(data), length(input_pop.neurons)))
        error("Dimensions of data must match number of neurons")
    end
    # If raw sensor data is 2D, the first dimension must represent space and second is time
    # Call generate_func for each vector of inputs in second dim, broadcast across the first dimension
    input_pop.generate_func.(input_pop.neurons, data)
    # Must sort the out_spikes of each InputNeuronPopulation after generating the spikes in reverse order based on time property of each spike
    sort!(input_pop.out_spikes, by=x->x.time, rev=true)
end

function process_next_spike!(cortex::Cortex)
    # Take the head spike from out_spikes queue of each population with smallest time property
    src_pop_id, spike = pop_next_spike!(cortex.populations)
    inds = findall(cortex.connectivity_matrix[:,src_pop_id])
    for i in inds
        # Route this spike to the correct populations with their weights using the process_spike function for every population that the spike is routed to
        src_pop = cortex.populations[i]
        weights = cortex.weights[src_pop_id=>i][:, spike.neuron_index]
        process_spike!(pop, weights, spike)
        # Find next_spike for each of the populations that just processed spikes and call output_spikes! to generate output spikes for each of those populations
        all_src_pop_ind = findall(cortex.connectivity_matrix[src_pop.id,:])
        _, next_spike, _ = get_next_spike(cortex.populations[all_src_pop_ind])
        output_spikes!(src_pop, next_spike)
        #  Call update_weights! to update the weights for those populations that just processed the spike
        update_weights!(src_pop, weights, spike, next_spike)
    end
end

function pop_next_spike!(pops::Array{NeuronPopulation, 1})
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

has_out_spikes(pop::NeuronPopulation) = num_out_spikes(pop) > 0

num_out_spikes(pop::NeuronPopulation) = length(pop.out_spikes)


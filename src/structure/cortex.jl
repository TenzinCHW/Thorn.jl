struct Cortex{T<:AbstractFloat}
    input_populations::Array{InputNeuronPopulation, 1}
    populations::Array{NeuronPopulation, 1}
    weights::Dict{Pair{Int, Int}, Array{T, 2}} # Input is along 2nd dimension
    connectivity_matrix::BitArray{2}

    # Constructor with connectivity_matrix and weight initialisation function
    function Cortex(input_neuron_types::Array{Tuple{UnionAll, Int}, 1}, neuron_types::Array{Tuple{UnionAll, Int, T, S}, 1}, connectivity_matrix::BitArray{2}, wt_init::Function, spiketype::UnionAll) where {T<:Any, S<:AbstractFloat}
        !(isequal(size(connectivity_matrix)...)) ? error("conn must be a square matrix") : nothing
        if !(isequal(first(size(connectivity_matrix)), length(input_neuron_types) + length(neuron_types)))
            error("length of one side of conn must be equal to sum of lengths of input_neuron_types and neuron_types")
        end
        !(spiketype<:Spike) ? error("spiketype must be a subtype of Spike") : nothing

        # Make the populations in the order given
        input_populations = InputNeuronPopulation[]
        for i in eachindex(input_neuron_types)
            push!(input_populations, InputNeuronPopulation(i, input_neuron_types[i]..., spiketype))
        end

        populations = NeuronPopulation[]
        num_inp_pop = length(input_populations)
        for i in eachindex(neuron_types)
            push!(populations, ProcessingNeuronPopulation(i + num_inp_pop, neuron_types[i]...))
        end
        populations = cat(input_populations, populations, dims=1)

        # Make the weights and insert into a Dict
        weights = Dict{Pair{Int, Int}, Array{typeof(wt_init()), 2}}()
        for j in 1:size(connectivity_matrix, 2)
            for i in 1:size(connectivity_matrix, 1)
                if (connectivity_matrix[i, j])
                    weights[j=>i] = wt_init(length(populations[i].neurons), length(populations[j].neurons))
                end
            end
        end
        # wt_init() must return a single value of the same type as the weights
        new{typeof(wt_init())}(input_populations, populations, weights, connectivity_matrix)
    end

    # Constructor with connectivity_matrix (use rand() for generating weights
    function Cortex(input_neuron_types::Array{Tuple{UnionAll, Int}, 1}, neuron_types::Array{Tuple{UnionAll, Int, T, S}, 1}, conn::BitArray{2}, spiketype::UnionAll) where {T<:Any, S<:AbstractFloat}
        Cortex(input_neuron_types, neuron_types, conn, rand, spiketype)
    end
end

function process_sample!(cortex::Cortex, input::Array{Array{T, 1}, 1}, maxval::T) where {T<:AbstractFloat}
    # Generate spike data from input; each array is for each corresponding input pop
    # If raw sensor data is 2D, flatten it
    for (pop, data) in zip(cortex.input_populations, input)
        generate_input_spikes!(pop, data, maxval)
    end
    # While any population in a Cortex has spikes in its out_spikes property, call process_next_spike
    while (any(length.(map(x->x.out_spikes, cortex.populations)) .> 0))
        process_next_spike!(cortex)
    end
    reset!(cortex)
end

function process_next_spike!(cortex::Cortex)
    # Take the head spike from out_spikes queue of each population with smallest time property
    src_pop_id, spike = pop_next_spike!(cortex.populations)
    inds = findall(cortex.connectivity_matrix[:, src_pop_id])
    for i in inds
        # Route this spike to the correct populations with their weights using the process_spike function for every population that the spike is routed to
        dst_pop = cortex.populations[i]
        weights = cortex.weights[src_pop_id=>i][:, spike.neuron_index]
        process_spike!(dst_pop, weights, spike)
        # Find next_spike for each of the populations that just processed spikes and call output_spike! to generate output spikes for each of those populations
        all_src_pop_ind = findall(cortex.connectivity_matrix[i, :])
        _, next_spike, _ = get_next_spike(cortex.populations[all_src_pop_ind])
        output_spike!(dst_pop, spike, next_spike)
        # TODO Find next spike among all the output spikes and then filter the new spikes based on their dependencies.
        #  Call update_weights! to update the weights for those populations that just processed the spike
        update_weights!(dst_pop, weights, spike, next_spike)
    end
end

function pop_next_spike!(pops::Array{NeuronPopulation, 1})
    non_empty_pops, _, ind = get_next_spike(pops)
    non_empty_pops[ind].id, pop!(non_empty_pops[ind].out_spikes)
end

function get_next_spike(pops::Array{NeuronPopulation, 1})
    non_empty_pops = filter(has_out_spikes, pops)
    if (isempty(non_empty_pops))
        return nothing, nothing, nothing
    end
    earliest_spikes = map(x->x.out_spikes[end], non_empty_pops)
    s, ind = findmin(earliest_spikes)
    non_empty_pops, s, ind
end

has_out_spikes(pop::NeuronPopulation) = num_out_spikes(pop) > 0

num_out_spikes(pop::NeuronPopulation) = length(pop.out_spikes)

reset!(cortex::Cortex) = reset!.(filter(x->isa(x, ProcessingNeuronPopulation), cortex.populations))

struct Cortex{T<:AbstractFloat}
    input_populations::Array{InputPopulation, 1}
    processing_populations::Array{ProcessingPopulation, 1}
    populations::Array{NeuronPopulation, 1}
    weights::Dict{Pair{Int, Int}, Array{T, 2}} # Input is along 2nd dimension
    connectivity_matrix::BitArray{2}

    # Constructor with connectivity_matrix and weight initialisation function
    function Cortex(input_neuron_types::Array{Tuple{UnionAll, Int}, 1}, neuron_types::Vector{Tuple{UnionAll, Int, T, S}}, connectivity::Vector{Pair{Int, Int}}, wt_init::Function, spiketype::UnionAll) where {T<:Any, S<:AbstractFloat}
        !(spiketype<:Spike) ? error("spiketype must be a subtype of Spike") : nothing

        # Make the populations in the order given
        input_populations = InputPopulation[]
        for i in eachindex(input_neuron_types)
            ntype, sz = input_neuron_types[i]
            push!(input_populations, ntype(i, sz, spiketype))
        end

        processing_populations = ProcessingPopulation[]
        num_inp_pop = length(input_populations)
        for i in eachindex(neuron_types)
            ntype, sz, weight_update, lr = neuron_types[i]
            push!(processing_populations, ntype(i + num_inp_pop, sz, weight_update, lr))
        end
        populations = vcat(input_populations, processing_populations)
        num_pop = length(populations)

        connectivity_matrix = BitArray(zeros(num_pop, num_pop))
        for (i, j) in connectivity
            connectivity_matrix[j, i] = true
        end

        # Make the weights and insert into a Dict
        weights = Dict{Pair{Int, Int}, Array{typeof(wt_init()), 2}}()
        for j in 1:size(connectivity_matrix, 2)
            for i in 1:size(connectivity_matrix, 1)
                if (connectivity_matrix[i, j])
                    weights[j=>i] = wt_init(populations[i].length, populations[j].length)
                end
            end
        end
        # wt_init() must return a single value of the same type as the weights
        new{typeof(wt_init())}(input_populations, processing_populations, populations, weights, connectivity_matrix)
    end

    # Constructor with connectivity_matrix (use rand() for generating weights
    function Cortex(input_neuron_types, neuron_types, connectivity, spiketype)
        Cortex(input_neuron_types, neuron_types, connectivity, rand, spiketype)
    end
end

# Assumes all inputs are normalized to 0. to 1.
function process_sample!(cortex::Cortex, input::Vector{Array{T, 2}},
                         maxval::T=1.,
                         extractors::Union{Dict, Nothing}=nothing) where {T<:AbstractFloat}
    # Reset all populations in cortex to ensure they're in correct state before processing
    reset!(cortex)
    # Generate spike data from input; each array is for each corresponding input pop
    # If raw sensor data is 2D, flatten it
    for (pop, data) in zip(cortex.input_populations, input)
        generate_input_spikes!(pop, data, maxval)
    end
    spikes, record = makerecord(extractors)
    # While any population in a Cortex has spikes in its out_spikes property, call process_next_spike
    total = 0
    while any(has_out_spikes.(cortex.populations))
        # Take the head spike from out_spikes queue of each population with smallest time property
        spike = process_next_spike(cortex)
        monitorrecord!(spikes, record, extractors, cortex, spike)
    end
    record = collapserecord(record)
    spikes, record
end

function process_next_spike(cortex)
    spike = pop_next_spike!(cortex.populations)
    process_spike!(cortex, spike)
    spike
end

function process_spike!(cortex::Cortex, spike::Spike)
    dst_pop_ids = dependent_populations(cortex, spike.pop_id)
    S_dst = Dict{Int, Vector{Spike}}()
    for i in dst_pop_ids
        # Route this spike to the correct populations with their weights using the process_spike function for every population that the spike is routed to
        dst_pop = cortex.populations[i]
        @views weights = cortex.weights[spike.pop_id=>i][:, spike.neuron_id]
        update_state!(dst_pop, weights, spike)
        # Find next_spike for each of the populations that just processed spikes and call output_spike! to generate output spikes for each of those populations
        S_dst[i] = output_spike!(dst_pop, spike)
    end
    #TODO find earliest spike of all populations (including new ones) and sort
    S_earliest = Spike[]
    for (i, pop) in enumerate(cortex.populations)
        if i in keys(S_dst)
            newspikes = S_dst[i]
            if !isempty(pop.out_spikes)
                if !isempty(newspikes)
                    push!(S_earliest, get_next_spike(pop, newspikes))
                else
                    push!(S_earliest, get_next_spike(pop))
                end
            else
                if !isempty(newspikes)
                    push!(S_earliest, get_next_spike(newspikes))
                end
            end
        end
    end
    #TODO loop over all earliest spikes and filter the new spikes produced by pops that each will go to
    for s in S_earliest
        for i in dst_pop_ids
            if spike.pop_id in population_dependency(cortex, i)
                filter!(sp->sp.time<=s.time, S_dst[i])
            end
        end
    end
    for i in dst_pop_ids
        #TODO set the spike state for those neurons that spiked successfully
        dst_pop = cortex.populations[i]
        update_spike!(dst_pop, S_dst[i])
        @views weights = cortex.weights[spike.pop_id=>i][:, spike.neuron_id]
        #TODO update weights for output spike each dst pop that happens before spike
        update_weights!(dst_pop, weights, spike)
        for ns in S_dst[i]
            dependency = population_dependency(cortex, i)
            for pop in cortex.populations[dependency]
                update_weights!(pop, dst_pop, cortex.weights[pop.id=>ns.pop_id], ns)
            end
            #TODO insert new spikes into their respective pops' out_spikes fields
            insertsorted!(dst_pop.out_spikes, ns, (x,y)->x.time<y.time)
        end
    end
end

function pop_next_spike!(pops::Vector{NeuronPopulation})
    spike = get_next_spike(pops)
    pop!(pops[spike.pop_id].out_spikes)
end

function get_next_spike(pops::Vector{NeuronPopulation})
    non_empty_pops = filter(has_out_spikes, pops)
    if isempty(non_empty_pops)
        return nothing
    end
    earliest_spikes = [get_next_spike(pop) for pop in non_empty_pops]
    _, ind = findmin(timing.(earliest_spikes))
    earliest_spikes[ind]
end

function get_next_spike(pop::NeuronPopulation, newspikes::Vector{Spike})
    popspike = get_next_spike(pop)
    earliest_new = get_next_spike(newspikes)
    popspike.time < earliest_new.time ? popspike : earliest_new
end

function get_next_spike(pop::NeuronPopulation)
    first(pop.out_spikes)
end

function get_next_spike(spikes::Vector{Spike})
    _, ind = findmin(timing.(spikes))
    spikes[ind]
end

has_out_spikes(pop::NeuronPopulation) = num_out_spikes(pop) > 0

num_out_spikes(pop::NeuronPopulation) = length(pop.out_spikes)

reset!(cortex::Cortex) = reset!.(cortex.populations)

dependent_populations(c::Cortex, i::Int) = findall(c.connectivity_matrix[:, i])

population_dependency(c::Cortex, i::Int) = findall(c.connectivity_matrix[i, :])

timing(s::Spike) = s.time

function makerecord(extractors::Union{Dict, Nothing})
    if !isnothing(extractors)
        record = Dict(k=>Array{Dict, 1}() for (k, _) in extractors)
        spikes = Spike[]
        return spikes, record
    end
    nothing, nothing
end

function monitorrecord!(spikes, record, extractors, cortex, spike)
    if !isnothing(record)
        monitor!(record, cortex, spike, extractors)
        push!(spikes, spike)
    end
end

function collapserecord(record)
    if !isnothing(record)
        record = Dict(k=>collapse(v) for (k, v) in record)
        return record
    end
end


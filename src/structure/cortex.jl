struct Cortex{T<:AbstractFloat}
    input_populations::Vector{InputPopulation}
    processing_populations::Vector{ProcessingPopulation}
    populations::Vector{NeuronPopulation}
    weights::Dict{Pair{Int, Int}, Weights} # Input is along 2nd dimension
    connectivity_matrix::BitArray{2}
    train_matrix::BitArray{2} # describes whether a connection should be trained
    S_earliest::Vector{Spike} # temp vector for holding sorted seq of earliest spikes from each pop to filter spike proposals
    S_proposed::Dict{Int, Vector{Spike}} # temp dict for holding vectors of spike proposals for each pop after input spike enters pop

    # Constructor with connectivity_matrix and weight initialisation function
    function Cortex(
            input_neuron_types::Vector,
            neuron_types::Vector,
            connectivity::Vector{Pair{Int, Int}},
            train_conn::Vector{Pair{Int, Int}},
            wt_init::Function, spiketype::UnionAll) where {T<:Any, S<:AbstractFloat}
        !(spiketype<:Spike) ? error("spiketype must be a subtype of Spike") : nothing

        input_populations = make_inp_pops(input_neuron_types, spiketype)
        num_inp_pop = length(input_populations)
        processing_populations = make_proc_pops(neuron_types, num_inp_pop)

        populations = vcat(input_populations, processing_populations)
        num_pop = length(populations)

        connectivity_matrix = BitArray(zeros(num_pop, num_pop))
        for (i, j) in connectivity
            connectivity_matrix[j, i] = true
        end

        train_matrix = BitArray(zeros(num_pop, num_pop))
        for (i, j) in train_conn
            train_matrix[j, i] = true
        end

        # Make the weights and insert into a Dict
        weights = Dict{Pair{Int, Int}, Weights}()
        for j in 1:size(connectivity_matrix, 2)
            for i in 1:size(connectivity_matrix, 1)
                if (connectivity_matrix[i, j])
                    w = wt_init(populations[i].length, populations[j].length)
                    weights[j=>i] = Weights(w)
                end
            end
        end
        S_earliest = Spike[]
        S_proposed = Dict{Int, Vector{Spike}}(pop.id=>Spike[] for pop in populations)
        # wt_init() must return a single value of the same type as the weights
        new{typeof(wt_init())}(
            input_populations,
            processing_populations,
            populations,
            weights,
            connectivity_matrix,
            train_matrix,
            S_earliest,
            S_proposed)
    end

    # Constructor with connectivity_matrix (use rand() for generating weights
    function Cortex(input_neuron_types, neuron_types, connectivity, spiketype)
        Cortex(input_neuron_types, neuron_types, connectivity, rand, spiketype)
    end

    function Cortex(
            input_neuron_types, neuron_types, connectivity, wt_init, spiketype)
        train_conn = connectivity[:]
        Cortex(input_neuron_types, neuron_types, connectivity, train_conn, wt_init, spiketype)
    end
end

function make_inp_pops(input_neuron_types::Vector, spiketype::UnionAll)
    # Make the populations in the order given
    input_populations = InputPopulation[]
    for i in eachindex(input_neuron_types)
        if length(input_neuron_types[i]) == 3
            ntype, sz, kwargs = input_neuron_types[i]
        else
            ntype, sz = input_neuron_types[i]
            kwargs = Dict()
        end
        push!(input_populations, ntype(i, sz, spiketype; kwargs...))
    end
    input_populations
end

function make_proc_pops(neuron_types::Vector, num_inp_pop::Int)
    processing_populations = ProcessingPopulation[]
    for i in eachindex(neuron_types)
        if length(neuron_types[i]) == 5
            ntype, sz, weight_update, lr, kwargs = neuron_types[i]
        else
            ntype, sz, weight_update, lr = neuron_types[i]
            kwargs = Dict()
        end
        push!(processing_populations, ntype(i + num_inp_pop, sz, weight_update, lr; kwargs...))
    end
    processing_populations
end

function freeze_weights!(cortex::Cortex, conn::Pair{Int, Int})
    freeze_unfreeze_weights!(cortex, conn, false)
end

function unfreeze_weights!(cortex::Cortex, conn::Pair{Int, Int})
    freeze_unfreeze_weights!(cortex, conn, true)
end

function freeze_unfreeze_weights!(cortex, conn::Pair{Int, Int}, val::Bool)
    i, j = conn
    cortex.train_matrix[j, i] = val
end

# Assumes all inputs are normalized to 0. to 1.
function process_sample!(
        cortex::Cortex,
        input::Vector{Array{T, 2}},
        maxval::T=1.;
        extractors::Union{Dict, Nothing}=nothing,
        train=true,
        reset_state=true) where {T<:AbstractFloat}
    # Reset all populations in cortex to ensure they're in correct state before processing
    reset!(cortex, reset_state)
    # Generate spike data from input; each array is for each corresponding input pop
    # If raw sensor data is 2D, flatten it
    # TODO consider allowing any shape of input data as long as it matches input pop
    for (pop, data) in zip(cortex.input_populations, input)
        generate_input_spikes!(pop, data, maxval)
    end
    record = isnothing(extractors) ? nothing : DefaultDict(()->DefaultDict(()->[]))
    # While any population has unprocessed spikes, call process_next_spike!
    while any(has_out_spikes.(cortex.populations))
        # Take the head spike from out_spikes queue of each population with smallest time property
        spike = process_next_spike!(cortex, train)
        monitorrecord!(record, cortex, spike, extractors)
    end
    record = collapserecord!(record)
    record
end

function process_next_spike!(cortex::Cortex, train::Bool)
    spike = pop_next_spike!(cortex.populations)
    process_spike!(cortex, spike, train)
    spike
end

function process_spike!(cortex::Cortex, spike::Spike, train::Bool)
    dst_pop_ids = dependent_populations(cortex, spike.pop_id)
    propagatespikecollectoutput!(cortex, spike, dst_pop_ids)
    nextspikebypop!(cortex, dst_pop_ids)
    filterafterearliest!(cortex, dst_pop_ids)
    outputspikesandupdateweights!(cortex, spike, dst_pop_ids, train)
    emptyallarr!(cortex.S_proposed)
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

function propagatespikecollectoutput!(
        cortex::Cortex, spike::Spike, dst_pop_ids::Vector{Int})
    for i in dst_pop_ids
        # Route this spike to the correct populations with their weights
        dst_pop = cortex.populations[i]
        weights = cortex.weights[spike.pop_id=>i]
        recvspike!(dst_pop, cortex.S_proposed[i], weights.value, spike)
        sort!(cortex.S_proposed[i], by=x->x.time)
    end
end

function nextspikebypop!(cortex::Cortex, dst_pop_ids::Vector{Int})
    # find each population's earliest spike which might cancel proposed spikes
    for pop in cortex.populations
        for dst_id in dst_pop_ids
            if pop.id in population_dependency(cortex, dst_id)
                newspikes = cortex.S_proposed[pop.id]
                if !isempty(pop.out_spikes)
                    if !isempty(newspikes)
                        push!(cortex.S_earliest, get_next_spike(pop, newspikes))
                    else
                        push!(cortex.S_earliest, get_next_spike(pop))
                    end
                else
                    if !isempty(newspikes)
                        push!(cortex.S_earliest, get_next_spike(newspikes))
                    end
                end
            end
        end
    end
    sort!(cortex.S_earliest, by=s->s.time)
end

function get_next_spike(pop::NeuronPopulation, newspikes::Vector{Spike})
    popspike = get_next_spike(pop)
    earliest_new = get_next_spike(newspikes)
    popspike.time < earliest_new.time ? popspike : earliest_new
end

function get_next_spike(pop::NeuronPopulation)
    first(pop.out_spikes)
end

function get_next_spike(spikes::Vector{S}) where S<:Spike
    first(spikes)
end

has_out_spikes(pop::NeuronPopulation) = num_out_spikes(pop) > 0

num_out_spikes(pop::NeuronPopulation) = length(pop.out_spikes)

function reset!(cortex::Cortex, reset_state::Bool)
    if reset_state
        reset!.(cortex.populations)
    end
    clear!.(pop.out_spikes for pop in cortex.populations)
end

dependent_populations(c::Cortex, i::Int) = findall(c.connectivity_matrix[:, i])

population_dependency(c::Cortex, i::Int) = findall(c.connectivity_matrix[i, :])

timing(s::Spike) = s.time

function filterafterearliest!(cortex::Cortex, dst_pop_ids::Vector{Int})
    # filter out proposed spikes that occur after the next incoming spike
    for s in cortex.S_earliest
        for i in dst_pop_ids
            if s.pop_id in population_dependency(cortex, i)
                filter!(sp->sp.time<=s.time, cortex.S_proposed[i])
            end
        end
    end
    empty!(cortex.S_earliest)
end

function outputspikesandupdateweights!(
        cortex::Cortex, spike::Spike, dst_pop_ids::Vector{Int}, train::Bool)
    for i in dst_pop_ids
        # update state of populations for neurons that spiked successfully
        dst_pop = cortex.populations[i]
        updatevalidspikes!(dst_pop, cortex.S_proposed[i])
        if train && cortex.train_matrix[i, spike.pop_id]
            weights = cortex.weights[spike.pop_id=>i]
            # update weights between output spikes from dst_pop
            # that occur before current spike
            update_weights!(dst_pop, weights, spike)
        end
        dependency = population_dependency(cortex, i)
        for ns in cortex.S_proposed[i]
            if train
                for pop in cortex.populations[dependency]
                    if cortex.train_matrix[i, pop.id]
                        # update weights between new spikes from dst_pop
                        # and all previous incoming spikes
                        weights = cortex.weights[pop.id=>ns.pop_id]
                        update_weights!(pop, dst_pop, weights, ns)
                    end
                end
            end
            insertsorted!(dst_pop.out_spikes, ns, (x,y)->x.time<y.time)
        end
    end
end

function emptyallarr!(S_proposed::Dict{Int, Vector{Spike}})
    for (i, arr) in S_proposed
        empty!(arr)
    end
end


"""
    ```Cortex(
        input_neuron_types::Vector,
        neuron_types::Vector,
        connectivity::Vector{Tuple{Pair{Int, Int}, DataType, F, Dict{Symbol, A}}},
        [train_conn::Vector{Pair{Int, Int}}],
        spiketype::UnionAll) where {A, F<:Function}```

Constructs a datastructure that holds references to `InputNeuronPopulation`s, `ProcessingNeuronPopulation`s and `Weights`.
`input_neuron_types` is a `Vector` of `Tuple`s. Each `Tuple` consists of an `InputNeuronPopulation` type, an `Int` which describes the size of the population, and optionally a `Dict{Symbol, Any}` for key word arguments.
`neuron_types` is similarly formatted but with a `ProcessingNeuronPopulation` as the first element of each `Tuple` instead.
`connectivity` is a `Vector` of `Tuple`s. Each element consists of a `Pair{Int, Int}` which describe the indices of the  source and sink populations of a connection, a subtype of `Weights`, a function of two positive integers which initialises the weights between two populations and optionally a `Dict{Symbol, Any}` which describes the key word arguments for the weights.
`train_conn` is a `Vector` of `Pair{Int, Int}` which describes the weights to train. Each pair consists of the indices of the source and sink neuron populations.
`spiketype` is a subtype of `Spike`.

Use `process_sample!` to generate `Spike`s and train the `Weights`.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);
```
"""
struct Cortex{S<:Spike}
    input_populations::Vector{InputPopulation}
    processing_populations::Vector{ProcessingPopulation}
    populations::Vector{NeuronPopulation}
    weights::Dict{Pair{Int, Int}, W} where W<:Weights # Input is along 2nd dimension
    connectivity_matrix::BitArray{2}
    # describes whether a connection should be trained
    train_matrix::BitArray{2}
    # temp vector for holding sorted seq of earliest spikes from
    # each pop to filter spike proposals
    S_earliest::Vector{Spike}
    # temp dict for holding vectors of spike proposals for each
    # pop after input spike enters pop
    S_proposed::Dict{Int, Vector{Spike}}

    # Constructor with connectivity_matrix and weight initialisation function
    function Cortex(
            input_neuron_types::Vector,
            neuron_types::Vector,
            connectivity::Vector{Tuple{Pair{Int, Int}, DataType, F, Dict{Symbol, A}}},
            train_conn::Vector{Pair{Int, Int}},
            spiketype::UnionAll) where {A, F<:Function}
        !(spiketype<:Spike) ? error("spiketype must be a subtype of Spike") : nothing

        input_populations = make_inp_pops(input_neuron_types, spiketype)
        num_inp_pop = length(input_populations)
        processing_populations = make_proc_pops(neuron_types, num_inp_pop)
        populations = vcat(input_populations, processing_populations)
        num_pop = length(populations)

        train_matrix = makematrix(train_conn, num_pop)
        connectivity_matrix = makematrix(first.(connectivity), num_pop)
        #weights = Dict{Pair{Int, Int}, A}() where A<:Weights
        #for conn in connectivity
        #            end
        weights = Dict(makeweights(conn, populations) for conn in connectivity)

        S_earliest = spiketype[]
        S_proposed = Dict{Int, Vector{spiketype}}(pop.id=>spiketype[] for pop in populations)
        new{spiketype}(
            input_populations,
            processing_populations,
            populations,
            weights,
            connectivity_matrix,
            train_matrix,
            S_earliest,
            S_proposed)
    end

    function Cortex(
            input_neuron_types::Vector,
            neuron_types::Vector,
            connectivity::Vector{Tuple{Pair{Int, Int}, DataType, F}},
            train_conn::Vector{Pair{Int, Int}},
            spiketype::UnionAll) where F<:Function
        connectivity =
            [Tuple([conn..., Dict{Symbol, Any}()]) for conn in connectivity]
        Cortex(input_neuron_types, neuron_types, connectivity, train_conn, spiketype)
    end

    function Cortex(
            input_neuron_types::Vector,
            neuron_types::Vector,
            connectivity,
            spiketype::UnionAll)
        train_conn = first.(connectivity)
        Cortex(input_neuron_types, neuron_types, connectivity, train_conn, spiketype)
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
        if length(neuron_types[i]) == 3
            ntype, sz, kwargs = neuron_types[i]
        else
            ntype, sz = neuron_types[i]
            kwargs = Dict()
        end
        push!(processing_populations, ntype(i + num_inp_pop, sz; kwargs...))
    end
    processing_populations
end

function makeweights(conn, populations)
    if length(conn) == 4
        ij, wt_type, wt_init, wt_params = conn
    else
        ij, wt_type, wt_init = conn
        wt_params = Dict{Symbol, Any}()
    end
    i, j = ij
    weightval = wt_init(populations[j].length, populations[i].length)
    ij=>wt_type(weightval; wt_params...)
end

function makematrix(matval::Vector{Pair{Int, Int}}, numpop)
    matrix = BitArray(zeros(numpop, numpop))
    for (i, j) in matval
        matrix[j, i] = true
    end
    return matrix
end

"""
    freeze_weights!(cortex::Cortex, conn::Pair{Int, Int})

Freezes the weights between the `conn[1]` to `conn[2]` populations.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> cortex.train_matrix
2×2 BitArray{2}:
 0  0
 1  0

julia> freeze_weights!(cortex, 1=>2);

julia> cortex.train_matrix
2×2 BitArray{2}:
 0  0
 0  0
```
"""
function freeze_weights!(cortex::Cortex, conn::Pair{Int, Int})
    freeze_unfreeze_weights!(cortex, conn, false)
end

"""
    unfreeze_weights!(cortex::Cortex, conn::Pair{Int, Int})

Unfreezes the weights between the `conn[1]` to `conn[2]` populations.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> cortex.train_matrix
2×2 BitArray{2}:
 0  0
 0  0

julia> unfreeze_weights!(cortex, 1=>2);

julia> cortex.train_matrix
2×2 BitArray{2}:
 0  0
 1  0
```
"""
function unfreeze_weights!(cortex::Cortex, conn::Pair{Int, Int})
    freeze_unfreeze_weights!(cortex, conn, true)
end

function freeze_unfreeze_weights!(cortex, conn::Pair{Int, Int}, val::Bool)
    i, j = conn
    cortex.train_matrix[j, i] = val
end

"""
    process_sample!(cortex::Cortex, input::Vector{Array{T, 2}}, maxval::T=1.; extractors::Union{Dict, Nothing}=nothing, train::Bool=true, reset_state::Bool=true)

Processes `input` using `cortex` to produce `Spikes`.
`cortex` is a `Cortex`.
`input` is a `Vector` of 2D `Array`s. The `Array` at index `i` is the input to the `i`-th `InputNeuronPopulation`. The first dimension must be equal to the population's `length` property while the second dimention is the time axis. So the `j, k`-th element of the `i`-th `Array` is the "rate" value for neuron `j` of the `i`-th population at time step `k`.
`maxval` is a normalising value, as the "rate" should be interpreted as being ∈ [0, `maxval`].
`extractors` is an optional key word argument, when specified should be a `Dict{String, F} where F<:Function`. The `String` is a user-defined name for the data which the `Function` extracts after every `Spike` is processed. See `monitor!` for the function specification.
`train` is a `Bool` which describes whether the weights of the `cortex` should be updated.
`reset_state` is a `Bool` which describes whether to reset the state of the `cortex` after processing all the spikes.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);

julia> data = [rand(inpsz, 2)];

julia> extractors = Dict("spikes"=>getspikes!);

julia> record = process_sample!(cortex, data; extractors=extractors);

julia> spikesfromrecord(record, "spikes")
7-element Array{Spike,1}:
 LIFSpike{Float64}(1, 3, 54.42308705519872, 1) 
 LIFSpike{Float64}(2, 10, 56.42308705519872, 1)
 LIFSpike{Float64}(2, 9, 56.42308705519872, 1) 
 LIFSpike{Float64}(2, 8, 56.42308705519872, 1) 
 LIFSpike{Float64}(2, 5, 56.42308705519872, 1) 
 LIFSpike{Float64}(1, 5, 93.98376027733754, 1) 
 LIFSpike{Float64}(1, 1, 163.90334909456485, 1)
```
"""
function process_sample!(
        cortex::Cortex,
        input::Vector{Array{T, 2}},
        maxval::T=1.;
        extractors::Union{Dict, Nothing}=nothing,
        train::Bool=true,
        reset_state::Bool=true) where {T<:AbstractFloat}
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
    propagatespike!(cortex, spike, dst_pop_ids)
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
    _, ind = findmin(timing(earliest_spikes))
    earliest_spikes[ind]
end

function propagatespike!(
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
        nextspike = get_next_spike(pop, cortex.S_proposed[pop.id])
        if !isnothing(nextspike)
            push!(cortex.S_earliest, nextspike)
        end
    end
    sort!(cortex.S_earliest, by=s->s.time)
end

function get_next_spike(pop::NeuronPopulation, newspikes::Vector{Spike})
    nextpopspike = get_next_spike(pop)
    nextnewspike = get_next_spike(newspikes)
    if !isnothing(nextpopspike)
        if !isnothing(nextnewspike)
            if nextpopspike.time < nextnewspike.time
                return nextpopspike
            else
                return nextnewspike
            end
        else
            return nextpopspike
        end
    end
end

function get_next_spike(pop::NeuronPopulation)
    get_next_spike(pop.out_spikes)
end

function get_next_spike(spikes)
    !isempty(spikes) ? first(spikes) : nothing
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
timing(s::Vector{S}) where S<:Spike = timing.(s)

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


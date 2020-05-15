import DataStructures.DefaultDict

"""
    getspikes!(record, cortex::Cortex, spike::S) where S<:Spike

An extractor for collecting spikes produced by running a sample through `cortex` with `process_sample!`.
See `spikesfromrecord` for extracting spikes from the record produced by `process_sample!`.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> extractors = Dict("userdefinedextractorname"=>getspikes!)
Dict{String,typeof(getspikes!)} with 1 entry:                                                                                                                                          
  "spikes" => getspikes!
```
"""
function getspikes!(record, cortex::Cortex, spike::S) where S<:Spike
    push!(record["spikes"], (spike, [nothing]))
end

"""
    spikesfromrecord(record, extractorname::String)::Vector{Spike}

Extracts the spikes from a record returned by `process_sample!` using `extractorname` that was used to name the `getspikes!` extractor. If no spikes were produced, this function will return an empty array.

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
function spikesfromrecord(record, extractorname::String)::Vector{Spike}
    # If no spikes are produced then it won't appear in the record
    if extractorname in keys(record)
        s = record[extractorname]
        return first(s["spikes"])
    else
        return []
    end
end

"""
    getu!(record, cortex::Cortex, spike::S) where S<:Spike

An extractor for collecting the `u` field (state) of each population. Extracts data as a `Dict` with keys being the population IDs and values being a 2-`Tuple` of a `Vector` of spikes and a 2D `Array` of `u` values of the poulation at the time of each spike. The first dimension of the 2D `Array` will be the same size as the size of the population, while the second dimension is the same as the number of spikes.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);

julia> data = [rand(inpsz, 2)];

julia> extractors = Dict("u"=>getu!);

julia> record = process_sample!(cortex, data; extractors=extractors);

julia> procpop = cortex.processing_populations[1];

julia> spikes, u = record["u"][procpop.id];

julia> println(length(spikes)); println(length(procpop.u)); println(size(u))
3
10
(10, 3)
```
"""
function getu!(record, cortex::Cortex, spike::S) where S<:Spike
    dependent_ids = dependent_populations(cortex, spike.pop_id)
    for pop_id in dependent_ids
        push!(record[pop_id], (spike, cortex.populations[pop_id].u[:]))
    end
end

"""
    getweights!(record, cortex::Cortex, spike::S) where S<:Spike

An extractor for collecting the `value` field of each weight in `cortex`. Extracts data as a `Dict` with keys being the pair of population IDs that the weight connects and values being a 2-`Tuple` of a `Vector` of spikes and an `Array` of weight values of the weights at the time of each spike. The first dimensions of the `Array` of weights will be the same as the size of the corresponding weights in `cortex`, while the last dimension is the same as the number of spikes.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);

julia> data = [rand(inpsz, 2)];

julia> extractors = Dict("weights"=>getweights!);

julia> record = process_sample!(cortex, data; extractors=extractors);

julia> println(length(spikes)); println(size(cortex.weights[1=>2].value)); println(size(weights))
3
(10, 5)
(10, 5, 3)
```
"""
function getweights!(record, cortex::Cortex, spike::S) where S<:Spike
    dependent_ids = dependent_populations(cortex, spike.pop_id)
    for (p2p, w) in cortex.weights
        if last(p2p) in dependent_ids
            push!(record[p2p], (spike, deepcopy(w.value)))
        end
    end
end

"""
    getvarfrompop(popid::Int, extractor::Function)

Pass in an extractor to run only when a spike whose `pop_id` field is equal to `popid` is processed.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz), (LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand), (2=>3, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);

julia> data = [rand(inpsz, 8)];

julia> extractors = Dict("getuwhen2spikes"=>getvarfrompop(2, getu!));

julia> record = process_sample!(cortex, data; extractors=extractors);

julia> getuwhen2fires = record["getuwhen2spikes"];

julia> spikes = first.(values(getuwhen2fires));

julia> all(all(s.pop_id == 2 for s in sp) for sp in spikes)
true
```
"""
function getvarfrompop(popid::Int, extractor::Function)
    function getvar!(record, cortex::Cortex, spike::S) where S<:Spike
        if spike.pop_id == popid
            extractor(record, cortex, spike)
        end
    end
end

function monitorrecord!(
        record::Union{DefaultDict, Nothing}, cortex::Cortex, spike::Spike, extractors)
    if !isnothing(record)
        monitor!(record, cortex, spike, extractors)
    end
end

function monitor!(record::DefaultDict, cortex::Cortex, spike::Spike, extractors)
    for (name, extractor) in extractors
        extractor(record[name], cortex, spike)
    end
end

function collapse(d::DefaultDict)
    result = Dict()
    for (k, v) in d
        spikes = first.(v)
        data = last.(v)
        data = cat(data..., dims=length(size(first(data)))+1)
        result[k] = (spikes, data)
    end
    result
end

function collapserecord!(record)
    if !isnothing(record)
        collapsed = Dict(k=>collapse(v) for (k, v) in record)
        return collapsed
    end
end


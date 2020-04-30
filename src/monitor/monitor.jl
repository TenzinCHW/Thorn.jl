import DataStructures.DefaultDict

function getspike!(record, cortex, spike)
    push!(record["spikes"], (spike, [nothing]))
end

function spikesfromrecord(record, extractorname::String)::Vector{Spike}
    if extractorname in keys(record) # If no spikes are produced then it won't appear in the record
        s = record[extractorname]
        return first(s["spikes"])
    else
        return []
    end
end

function getu!(record, cortex::Cortex, spike::S) where S<:Spike
    dependent_ids = dependent_populations(cortex, spike.pop_id)
    for pop_id in dependent_ids
        push!(record[pop_id], (spike, cortex.populations[pop_id].u[:]))
    end
end

function getweights!(record, cortex::Cortex, spike::S) where S<:Spike
    dependent_ids = dependent_populations(cortex, spike.pop_id)
    for (p2p, w) in cortex.weights
        if last(p2p) in dependent_ids
            push!(record[p2p], (spike, deepcopy(w.value)))
        end
    end
end

function getvarfrompop(popid::Int, extractor::Function)
    function getvar!(recordcortexspike...)
        if spike.pop_id == popid
            extractor(recordcortexspike...)
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


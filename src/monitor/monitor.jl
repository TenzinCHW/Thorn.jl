import DataStructures.DefaultDict

function monitorrecord!(record::Union{DefaultDict, Nothing}, cortex::Cortex, spike::Spike, extractors)
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


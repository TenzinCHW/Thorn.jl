function monitor!(cortex::Cortex, pop_spike::Tuple{UInt, S}, extractors::Dict, data::Dict{String, Array{Dict, 1}}) where S<:Spike
    for (name, extractor) in extractors
        push!(data[name], deepcopy(extractor(cortex, pop_spike)))
    end
end

function collapse(arr::Array{Dict, 1})
    ks = collect(keys(first(arr)))
    result = Dict(key=>[] for key in ks)
    for t in arr
        map(k->push!(result[k], t[k]), filter(k->!isnothing(t[k]), ks))
    end
    collapse(result)
end

function collapse(d::Dict)
    result = Dict()
    for (k, v) in d
        result[k] = cat(v..., dims=length(size(first(v)))+1)
    end
    result
end


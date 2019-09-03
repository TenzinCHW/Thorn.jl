function monitor!(record::Dict{String, Array{Dict, 1}}, cortex::Cortex, spike::Spike, extractors::Dict)
    for (name, extractor) in extractors
        push!(record[name], deepcopy(extractor(cortex, spike)))
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


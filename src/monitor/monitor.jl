#struct Monitor
#    extractors::Dict{String, Function}
#    data::Dict{String, Array}
#
#    function Monitor(extractors::Dict{String, Function})
#        data = Dict(name=>[] for (name, _) in extractors)
#        new(extractors, data)
#    end
#end

function monitor(cortex::Cortex, extractors::Dict, data::Dict{String, Array{Dict, 1}})
    for (name, extractor) in extractors
        push!(data[name], deepcopy(extractor(cortex)))
    end
end

function collapse(arr::Array{Dict, 1})
    ks = collect(keys(first(arr)))
    result = Dict(key=>[] for key in ks)
    for t in arr
        map(k->push!(result[k], t[k]), ks)
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


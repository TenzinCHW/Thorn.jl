import Random

mutable struct Dataset
    datasrc::Datasource
    classes::Array{String, 1}
    traintestdir::String
    data::Dict{String, Array{String, 1}}
    activeset::Array{String, 1}

    function Dataset(basedir::String, traintestdir::String, activecls::Array{String, 1}=["all"])
        src = Datasource(basedir)
        data, activeset, classes = getdsdata(src, traintestdir, activecls)
        new(src, classes, traintestdir, data, activeset)
    end
end

function getdsdata(src::Datasource, traintestdir::String, activecls::Array{String, 1})
    classes = datasrcitems(src, traintestdir)
    if activecls == ["all"]
        activecls = classes
    end
    all(in.(activecls, [classes])) || error("Invalid active class")
    data = Dict(cls => datasrcitems(src, joinpath(traintestdir, cls))
                    for cls in classes)
    activeset = sourcenames(data, activecls)
    return data, activeset, activecls
end

function swaptraintest!(dataset::Dataset, traintestdir::String)
    dataset.traintestdir = traintestdir
    dataset.data, dataset.activeset, dataset.classes = getdsdata(dataset.datasrc,
                                                            traintestdir,
                                                            dataset.classes)
end

function resizeset!(dataset::Dataset, percentage::Int64)
    percentage < 1 || percentage > 100 &&
    error("percentage should be from 1 to 100")
    frac = 100 / percentage
    len = Int.(ceil.(length.(values(dataset.data))./frac))
    data = Dict{String, Array{String, 1}}()
    for (i, (k, v)) in enumerate(dataset.data)
        data[k] = v[1:i]
    end
    dataset.activeset = sourcenames(data, dataset.classes)
end

# Gets class/filename as an array
function sourcenames(data::Dict{String, Array{String, 1}},
                                               activecls::Array{String, 1})
    activedata = filter(cls_d->first(cls_d) in activecls, data)
    setnames = []
    for entry in activedata
        push!(setnames, joinpath.(entry...))
    end
    collect(Iterators.flatten(setnames))
end

function shufflebyclass!(dataset::Dataset)
    for cls in dataset.classes
        Random.shuffle!(dataset.data[cls])
    end
end

function Base.getindex(dataset::Dataset, cls::String, i::Int64)
    (i < 1 || !(cls in dataset.classes)) &&
    error("index $i cannot be less than 1 and $cls must be valid active class")
    i > length(dataset.data[cls]) &&
    error("index i > $(length(dataset.data[cls])) (number of class $(dataset.classes[cls]) examples)")
    fname = dataset.data[cls][i]
    path = (dataset.traintestdir, cls, fname)
    readfrompath(dataset.datasrc, path...)
end

Base.iterate(dataset::Dataset, i=1) = i > length(dataset.activeset) ? nothing : getnext(dataset, i)

function getnext(dataset::Dataset, i)
    fpath = (dataset.traintestdir, dataset.activeset[i])
    readfrompath(dataset.datasrc, fpath...), i + 1
end

Base.length(dataset::Dataset) = length(dataset.activeset)

readfrompath(src::Datasource, path::String...) = datasrcread(src, joinpath(path...))

mutable struct Dataloader
    dataset::Dataset
    shuffle::Bool
    transform::Function

    function Dataloader(dataset, shuffle, transform)
        new(dataset, shuffle, transform)
    end

    function Dataloader(dataset, shuffle)
        Dataloader(dataset, shuffle, x->x)
    end
end

function Base.iterate(loader::Dataloader, i=1)
    if i > length(loader) && loader.shuffle
        Random.shuffle!(loader.dataset.activeset)
    end
    item, j = iterate(loader.dataset, i)
    isnothing(item) ? nothing : loader.transform(item), j
end

Base.length(loader::Dataloader) = length(loader.dataset)


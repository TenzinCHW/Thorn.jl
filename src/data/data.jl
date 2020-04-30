import Random

mutable struct Dataset
    datasrc::Datasource
    classes::Vector{String}
    traintestdir::String
    data::Dict{String, Vector{String}}
    activeset::Vector{Tuple{String, String}}

    function Dataset(basedir::String, traintestdir::String; activecls::Vector{String}=["all"])
        src = Datasource(basedir)
        data, activeset, classes = getdsdata(src, traintestdir, activecls)
        new(src, classes, traintestdir, data, activeset)
    end
end

function getdsdata(src::Datasource, traintestdir::String, activecls::Vector{String})
    classes = datasrcitems(src, traintestdir)
    if activecls == ["all"]
        activecls = classes
    end
    all(in.(activecls, [classes])) || error("Invalid active class")
    data = Dict(cls => datasrcitems(src, joinpath(traintestdir, cls))
                    for cls in activecls)
    activeset = sourcenames(data, activecls)
    return data, activeset, activecls
end

function swaptraintest!(dataset::Dataset, traintestdir::String)
    dataset.traintestdir = traintestdir
    dataset.data, dataset.activeset, dataset.classes =
        getdsdata(
            dataset.datasrc,
            traintestdir,
            dataset.classes)
end

function resizeset!(dataset::Dataset, percentage::Int64)
    percentage < 1 || percentage > 100 &&
    error("percentage should be from 1 to 100")
    frac = 100 / percentage
    len = Int.(ceil.(length.(values(dataset.data))./frac))
    data = Dict{String, Vector{String}}()
    for (n, (k, v)) in zip(len, dataset.data)
        data[k] = v[1:n]
    end
    dataset.activeset = sourcenames(data, dataset.classes)
end

# Gets class/filename as an array
function sourcenames(
        data::Dict{String, Vector{String}}, activecls::Vector{String})
    activedata = filter(cls_d->first(cls_d) in activecls, data)
    setnames = []
    for entry in activedata
        append!(setnames, class_datapath.(entry...))
    end
    setnames
end

class_datapath(cls::String, path::String) = (cls, path)

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
    fpath = (dataset.traintestdir, dataset.activeset[i]...)
    item = first(dataset.activeset[i]), readfrompath(dataset.datasrc, fpath...)
    item, i + 1
end

Base.length(dataset::Dataset) = length(dataset.activeset)

readfrompath(src::Datasource, path::String...) = datasrcread(src, joinpath(path...))

mutable struct Dataloader
    dataset::Dataset
    shuffle::Bool
    transform::Function

    function Dataloader(dataset, shuffle, transform=x->x)
        new(dataset, shuffle, transform)
    end
end

function Base.iterate(loader::Dataloader, i=1)
    if i > length(loader) && loader.shuffle
        Random.shuffle!(loader.dataset.activeset)
    end
    item = iterate(loader.dataset, i)
    if isnothing(item)
        return item
    else
        (cls, data), j = item
        return ((cls, loader.transform(data)), j)
    end
end

Base.length(loader::Dataloader) = length(loader.dataset)


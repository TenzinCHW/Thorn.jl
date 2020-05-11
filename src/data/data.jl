import Random

"""
    `Dataset(src::String, partition::String; activecls::Vector{String}=["all"])`

Constructs a datastructure for holding references to data file paths.
`src` is the name of the dataset file.
`partition` is the name of the partition of the dataset to use
i.e. "train", "val" and "test". Iterating over the dataset will only iterate
over data from this partition.
`activecls` is an array of the class names to use. By default it will use all
classes.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> dataset = Dataset("example.jld2", "train")
output omitted

julia> for (k, v) in dataset
           println(k, " ", v)
       end
class1 0.22927480568219583
class1 0.7178734191741023
class1 0.9123544607857246
class1 0.11874593984889614
class2 0.7142581931358412
class2 0.8067948527443549
class2 0.10183891896498287
class2 0.04750070239079185

julia> dataset = Dataset("example.jld2", "train"; activecls=["class2"])
output omitted

julia> for (k, v) in dataset
           println(k, " ", v)
       end
class2 0.7142581931358412
class2 0.8067948527443549
class2 0.10183891896498287
class2 0.04750070239079185
```
"""
struct Dataset
    datasrc::Datasource
    classes::Vector{String}
    partition::Array{String, 0}
    data::Dict{String, Vector{String}}
    activeset::Vector{Tuple{String, String}}

    function Dataset(src::String, partition::String; activecls::Vector{String}=["all"])
        src = Datasource(src)
        data, activeset, classes = getdsdata(src, partition, activecls)
        partitionarr = Array{String}(undef)
        partitionarr[] = partition
        new(src, classes, partitionarr, data, activeset)
    end
end

function getdsdata(src::Datasource, partition::String, activecls::Vector{String})
    classes = datasrcitems(src, partition)
    if activecls == ["all"]
        activecls = classes
    end
    all(in.(activecls, [classes])) || error("Invalid active class")
    data = Dict(cls => datasrcitems(src, joinpath(partition, cls))
                    for cls in activecls)
    activeset = sourcenames(data, activecls)
    return data, activeset, activecls
end

"""
    `swappartition!(dataset::Dataset, partition::String)`

Changes the partition of the dataset to use.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> dataset.partition
0-dimensional Array{String,0}:
"train"

julia> swappartition!(dataset, "val")

julia> dataset.partition
"val"
```
"""
function swappartition!(dataset::Dataset, partition::String)
    dataset.partition[] = partition
    data, activeset, _ = getdsdata(
            dataset.datasrc,
            partition,
            dataset.classes)
    swap!(dataset.data, data)
    swap!(dataset.activeset, activeset)
    return nothing
end

function swap!(arr::Array, newitems::Array)
    empty!(arr)
    append!(arr, newitems)
end

function swap!(dict::Dict, newitems::Dict)
    empty!(dict)
    merge!(dict, newitems)
end

"""
    `resizeprop!(dataset::Dataset, percentage::Int64)`

Resizes the working set of data to be `percentage/100` the size of all
data in the current partition.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> for (k, v) in dataset
           println(k, " ", v)
       end
class1 0.22927480568219583
class1 0.7178734191741023
class1 0.9123544607857246
class1 0.11874593984889614
class2 0.7142581931358412
class2 0.8067948527443549
class2 0.10183891896498287
class2 0.04750070239079185

julia> resizeprop!(dataset, 40)

julia> for (k, v) in dataset
           println(k, " ", v)
       end
class1 0.22927480568219583
class1 0.7178734191741023
class2 0.7142581931358412
class2 0.8067948527443549
```
"""
function resizeprop!(dataset::Dataset, percentage::Int64)
    percentage < 1 || percentage > 100 &&
    error("percentage should be from 1 to 100")
    frac = 100 / percentage
    len = Int.(ceil.(length.(values(dataset.data))./frac))
    data = Dict{String, Vector{String}}()
    for (n, (k, v)) in zip(len, dataset.data)
        data[k] = v[1:n]
    end
    swap!(dataset.activeset, sourcenames(data, dataset.classes))
    return nothing
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

"""
    shufflebyclass!(dataset::Dataset)

Randomly shuffles data samples within each class in `dataset`.
This affects the order of iteration over `dataset.data[cls]`
for which `cls` is any element in `dataset.classes`.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> for cls in dataset.classes
           println(dataset.data[cls])
       end
["1", "2", "3", "4"]
["1", "2", "3", "4"]

julia> shufflebyclass!(dataset)

julia> for cls in dataset.classes
           println(dataset.data[cls])
       end
["3", "2", "4", "1"]
["3", "2", "1", "4"]
```
"""
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
    path = (dataset.partition[], cls, fname)
    readfrompath(dataset.datasrc, path...)
end

Base.iterate(dataset::Dataset, i=1) = i > length(dataset.activeset) ? nothing : getnext(dataset, i)

function getnext(dataset::Dataset, i)
    fpath = (dataset.partition[], dataset.activeset[i]...)
    item = first(dataset.activeset[i]), readfrompath(dataset.datasrc, fpath...)
    item, i + 1
end

Base.length(dataset::Dataset) = length(dataset.activeset)

readfrompath(src::Datasource, path::String...) = datasrcread(src, joinpath(path...))

"""
    `Dataloader(dataset::Dataset, shuffle::Bool=false, transform=x->x)`

A datastructure that holds a reference to a Dataset.
Controls iteration procedure such as whether to shuffle and transform the
data on iteration.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> dataloader = Dataloader(dataset)
output omitted

julia> for (cls, sample) in dataloader
           println(cls, " ", sample)
       end
class1 0.22927480568219583
class1 0.7178734191741023
class1 0.9123544607857246
class1 0.11874593984889614
class2 0.7142581931358412
class2 0.8067948527443549
class2 0.10183891896498287
class2 0.04750070239079185

julia> add5(data) = data .+ 5
add5 (generic function with 1 method)

julia> dataloader = Dataloader(dataset, true, add5)
output omitted

julia> for (cls, sample) in dataloader
           println(cls, " ", sample)
       end
class2 5.101838918964983
class1 5.229274805682196
class1 5.912354460785725
class1 5.7178734191741025
class2 5.714258193135841
class2 5.047500702390792
class1 5.118745939848896
class2 5.806794852744355
```
"""
mutable struct Dataloader
    dataset::Dataset
    shuffle::Bool
    transform::Function

    function Dataloader(dataset::Dataset, shuffle::Bool=false, transform=x->x)
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


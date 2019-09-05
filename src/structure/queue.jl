# This is an implementation of a queue whose elements are not deleted when popped
# i.e. you can still loop over all elements
# The underlying data structure is an array, with the first element being the head of the queue
# Popping increments the head index by 1 and returns the element at the (head-1)-th index
# Pushing just pushes into the end of the queue
# Inserting inserts into the array only if the insertion is after the head-th index
# Indexing the queue indexes directly into the array (able to access items that have been popped)

struct Queue{T}
    items::Vector{T}
    head::Array{Int, 0}

    function Queue(T)
        head = Array{Int}(undef)
        head[] = 1
        new{T}(T[], head)
    end
end

Base.length(q::Queue) = length(q.items) - q.head[] + 1
Base.firstindex(q::Queue) = q.head[]
Base.getindex(q::Queue, index::Int) = q.items[index]
Base.iterate(q::Queue) = isempty(q.items) ? nothing : iterate(q.items[q.head[]:end])
Base.iterate(q::Queue, i::Int) = iterate(q.items, i)
Base.iterate(q::Queue, n::Nothing) = iterate(q.items, n)

function Base.insert!(q::Queue{T}, index::Integer, item::T) where T
    if index < q.head[]
        error("Head of Queue is at $(q.head[]). Cannot insert before this index.")
    else
        insert!(q.items, index, item)
    end
end

function Base.push!(q::Queue{T}, item::T) where T
    push!(q.items, item)
end

function Base.pop!(q::Queue)
    q.head[] += 1
    q.items[q.head[] - 1]
end

#Base.Sort.searchsortedlast(s::Stack{T}, item::T; lt) where T = searchsortedlast(s.items, item, lt=lt)

function insertsorted!(q::Queue{T}, item::T, lt) where T
#    if isempty(out_spikes) || s.time <= out_spikes[end].time
#        push!(out_spikes, s)
#    else
        index = searchsortedfirst(q.items, item, lt=lt)
        insert!(q, index, item)
#    end
end

function clear!(q::Queue)
    for i in eachindex(q.items)
        pop!(q.items)
    end
    q.head[] = 1
end


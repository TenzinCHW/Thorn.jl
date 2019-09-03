#This is an implementation of a stack whose elements are not deleted when popped
#i.e. you can still loop over all elements

struct Stack{T}
    items::Vector{T}
    head::Array{Int, 0}

    function Stack(T)
        head = Array{Int}(undef)
        head[] = 0
        new{T}(T[], head)
    end
end

Base.length(s::Stack) = s.head[]
Base.lastindex(s::Stack) = s.head[]
Base.getindex(s::Stack, index::Int) = s.items[index]
Base.iterate(s::Stack) = isempty(s.items) ? nothing : iterate(s.items[1:s.head[]])
Base.iterate(s::Stack, n::Nothing) = iterate(s.items, n)

function Base.insert!(s::Stack{T}, index::Integer, item::T) where T
    if index > s.head[] + 1
        error("Stack has only $(s.head[]) elements. Cannot insert past $(s.head[] + 1)th element.")
    else
        insert!(s.items, index, item)
        s.head[] += 1
    end
end

function Base.push!(s::Stack{T}, item::T) where T
    insert!(s.items, s.head[] + 1, item)
    s.head[] += 1
end

function Base.pop!(s::Stack)
    res = s.items[s.head[]]
    s.head[] -= 1
    res
end

Base.Sort.searchsortedlast(s::Stack{T}, item::T; lt) where T = searchsortedlast(s.items, item, lt=lt)

function insertsorted!(s::Stack{T}, item::T) where T
    for s in sort(spikes, by=x->x.time, rev=true)
        if isempty(out_spikes) || s.time <= out_spikes[end].time
            push!(out_spikes, s)
        else
            index = searchsortedfirst(out_spikes, s, lt=(x,y)->x.time<y.time)
            insert!(pop.out_spikes, index, s)
        end
    end
end

function clear!(s::Stack)
    for i in eachindex(s.items)
        pop!(s.items)
    end
    s.head[] = 0
end


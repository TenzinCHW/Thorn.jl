function rasterspikes(spikes::Array{Tuple{UInt, T}, 1}, cortex::Cortex) where T<:Spike
    currentht = 0.
    dy = 0.1
    x = []
    y = []
    for (i, num) in enumerate(numneurons(cortex))
        sp = filter(x->first(x) == i, spikes)
        xs = [last(s).time for s in sp]
        ys = [currentht + last(s).neuron_index * dy for s in sp]
        push!(x, xs)
        push!(y, ys)
        currentht += 0.5
    end
    x, y
end

numneurons(cortex::Cortex) = [length(pop.neurons) for pop in cortex.populations]

function gridify(val_func::Array{Tuple{T, A}, 1}, spikes::Array{S, 1}, dt::T, time_end::T) where {T<:AbstractFloat, S<:Spike, A}
    # val_func is an array of tuples with first value as the i-th value to be plotted, second as function to compute the value between the (i-1)-th and i-th spike
    !isequal(length.([val_func, spikes])...) ? error("val_func should be same length as spikes") : nothing
    isfloat.(first.(val_func)) # Should not throw error
    x = makegrid(dt, time_end)
    y = T[]
    start = 0.
    for (s, func) in zip(spikes, last.(val_func))
        dy = func.(filter(dx->start<=dx<s.time, x) .- start)
        start = s.time
        for ddy in dy
            push!(y, ddy)
        end
    end
    x, y
end

function isfloat(item::AbstractFloat)
end

function makegrid(dt::T, et::T) where T<:AbstractFloat
    # dt is the resolution of the grid in ms
    [i*dt for i in 0:Int(ceil(et/dt))]
end


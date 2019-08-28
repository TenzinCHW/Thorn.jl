function rasterspikes(spikes::Array{Tuple{Int, T}, 1}, cortex::Cortex) where T<:Spike
    currentht = 0.
    dy = 1
    x = []
    y = []
    for (i, num) in enumerate(length.(cortex.populations))
        sp = filter(x->first(x) == i, spikes)
        xs = [last(s).time for s in sp]
        ys = [currentht + last(s).neuron_index * dy for s in sp]
        push!(x, xs)
        push!(y, ys)
        currentht += dy * (num + 1)
    end
    x, y
end

function gridify(val::Array{T, 1}, diffeq, spikes::Array{S, 1}, dt::T, time_end::T) where {T<:AbstractFloat, S<:Spike, A}
    # val is an array of initial values of the diffeq corresponding to the times of the spikes
    !isequal(length.([val, spikes])...) ? error("val should be same length as spikes") : nothing
    isfloat.(val) # Should not throw error
    x = makegrid(dt, time_end)
    y = T[]
    start = 0
    for (s, v) in zip(spikes, val)
        dy = diffeq.(v, filter(dx->start<=dx<s.time, x) .- start)
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


function rasterspikes(spikes::Vector{S}, cortex::Cortex) where S<:Spike
    currentht = 0.
    dy = 1
    x = Vector{typeof(currentht)}[]
    y = Vector{typeof(currentht)}[]
    for (i, num) in enumerate(pop.length for pop in cortex.populations)
        sp = filter(x->x.pop_id == i, spikes)
        xs = [s.time for s in sp]
        ys = [currentht + s.neuron_id * dy for s in sp]
        push!(x, xs)
        push!(y, ys)
        currentht += dy * (num + 1)
    end
    x, y
end

function gridify(val::Vector{T}, diffeq, spikes::Vector{S}, dt::T, time_end::T;
                 init_val::T=0.) where {T<:AbstractFloat, S<:Spike, A}
    # val is an array of initial values of the diffeq corresponding to the times of the spikes
    # spikes is an array of spikes
    !isequal(length(val), length(spikes)) ? error("val should be same length as spikes") : nothing
    isfloat.(val) # Should not throw error
    sort!(spikes, by=s->s.time)
    x = makegrid(dt, time_end)
    y = T[]
    start = 0.
    for (s, v) in zip(spikes, val)
        interpolate!(x, y, diffeq, init_val, start, s.time)
        init_val = v
        start = s.time
    end
    # Compute for the interval between time_end and the last spike timing
    interpolate!(x, y, diffeq, init_val, start, time_end + dt)
    x, y
end

function interpolate!(x::Array{T, 1}, y::Array{T, 1}, f::Function, u0::T,
                     start::T, time_end::T) where T<:AbstractFloat
    dy = f.(u0, filter(dx->start<=dx<time_end, x) .- start)
    for ddy in dy
        push!(y, ddy)
    end
end

function isfloat(item::AbstractFloat)
end

function makegrid(dt::T, et::T) where T<:AbstractFloat
    # dt is the resolution of the grid in ms
    [i*dt for i in 0:Int(ceil(et/dt))]
end


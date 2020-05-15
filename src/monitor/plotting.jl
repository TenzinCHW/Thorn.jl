"""
    rasterspikes(cortex::Cortex) where S<:Spike

Produces a `Tuple` of `x` and `y`, which are `Array`s representing points for plotting a raster of spikes produced by `cortex` since the last time it was passed through `process_sample!` with the `reset_state` parameter set to `true`.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);

julia> data = [rand(inpsz, 2)];

julia> process_sample!(cortex, data);

julia> x, y = rasterspikes(cortex)
(Array{Float64,1}[[54.42308705519872, 93.98376027733754, 163.90334909456485], [56.42308705519872, 56.42308705519872, 56.42308705519872, 56.42308705519872]], Array{Float64,1}[[3.0, 5.0, 1.0], [15.0, 14.0, 13.0, 10.0]])
```
"""
function rasterspikes(cortex::Cortex) where S<:Spike
    currentht = 0.
    dy = 1
    x = Vector{typeof(currentht)}[]
    y = Vector{typeof(currentht)}[]
    for pop in cortex.populations
        sp = pop.out_spikes.items
        xs = [s.time for s in sp]
        ys = [currentht + s.neuron_id * dy for s in sp]
        push!(x, xs)
        push!(y, ys)
        currentht += dy * pop.length
    end
    x, y
end


"""
    gridify(
        diffeq::Function,
        val::Vector{T},
        spikes::Vector{S},
        dt::T,
        time_end::T;
        init_val::T=0.) where {T<:AbstractFloat, S<:Spike}

Produces a `Tuple` of `x` and `y`, each of which are `Array`s representing the points of the curve described by a system governed by `diffeq` with fixed points at `collect(zip(val, timing(spikes))`.
`x` will start from 0. and end at `time_end`, with `dt` between each value of `x`.
`y` will start from `init_val` and will be the values interpolated using `diffeq` between each consecutive fixed point using the values in `x`.
`diffeq` must take the form of `diffeq(initial_val, dt)` where `initial_val` is the initial value and `dt` is the time step.

Examples
≡≡≡≡≡≡≡≡≡≡

```
julia> inpsz = 5; procsz = 10; spiketype = LIFSpike;

julia> inp_neuron_types = [(RateInpPopulation, inpsz)]; proc_neuron_types = [(LIFPopulation, procsz)];

julia> conn = [(1=>2, STDPWeights, rand)];

julia> cortex = Cortex(inp_neuron_types, proc_neuron_types, conn, spiketype);

julia> data = [rand(inpsz, 2)];

julia> extractors = Dict("u"=>getu!);

julia> record = process_sample!(cortex, data; extractors=extractors);

julia> procpop = cortex.processing_populations[1];

julia> diffeq = procpop.u_func;

julia> spikes, u = record["u"][procpop.id];

julia> x, y = gridify(diffeq, u[1, :], spikes, 1., 170.)
([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0  …  161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.005724444903867722, 0.005393646976814462, 0.005081964836598247, 0.24792104831915646, 0.23359445941244225, 0.2200957596708213, 0.20737710794563344, 0.1953934276799074, 0.18410224715117646, 0.17346354894616647])
```
"""
function gridify(
        diffeq::Function,
        val::Vector{T},
        spikes::Vector{S},
        dt::T,
        time_end::T;
        init_val::T=0.) where {T<:AbstractFloat, S<:Spike}
    # val is an array of initial values of the diffeq
    # corresponding to the times of the spikes
    !isequal(length(val), length(spikes)) && error("val should be same length as spikes")
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

function interpolate!(
        x::Array{T, 1},
        y::Array{T, 1},
        f::Function,
        u0::T,
        start::T,
        time_end::T) where T<:AbstractFloat
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


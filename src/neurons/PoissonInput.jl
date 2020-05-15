# Rates are in Hz
maxrate = 20.
minrate = 0.
sampleperiod = 100. # Milliseconds

"""
    ```PoissonInpPopulation(
        id, sz spiketype;
        maxrate=20., minrate=0., sampleperiod=100., sign=1)```

Datastructure for generating spikes via a Poisson process.
Use this type as input to `Cortex`. See `Cortex`.
"""
struct PoissonInpPopulation{T<:AbstractFloat, S<:Spike} <: InputPopulation
    id::Int
    maxrate::T
    minrate::T
    sampleperiod::T
    spiketype::UnionAll
    out_spikes::Queue{S}
    length::Int
    sign::Int8

    function PoissonInpPopulation(
            id,
            sz,
            spiketype;
            maxrate=maxrate,
            minrate=minrate,
            sampleperiod=sampleperiod,
            sign=1)
        (id < 1 || sz < 1) ? error("id and sz must be > 0") : nothing

        sign ∉ (-1, 1) && error("sign must be 1 or -1")
        out_spikes = Queue(spiketype)
        new{typeof(maxrate), spiketype}(
            id, maxrate, minrate, sampleperiod, spiketype, out_spikes, sz, sign)
    end
end

function generate_spikes(
        pop::PoissonInpPopulation{T},
        neuron_id::Int,
        sensor_inp::Vector{T},
        maxval::T) where T<:AbstractFloat
    rate = compute_rate.(pop, maxval, sensor_inp)
    expacc = ExpAccumulate.(rate, pop.sampleperiod)
    spikes = pop.spiketype[]
    start = 0.
    for acc in expacc
        if sign(acc.rate) == pop.sign
            for t in acc
                push!(spikes, pop.spiketype(pop.id, neuron_id, t + start, pop.sign))
            end
        end
        start += pop.sampleperiod
    end
    spikes
end

reset!(p::PoissonInpPopulation) = nothing

mutable struct ExpAccumulate{T<:AbstractFloat}
    t::T
    rate::T
    max::T

    function ExpAccumulate(rate, max)
        T = typeof(rate)
        new{T}(T(0), rate, max)
    end
end

function Base.iterate(e::ExpAccumulate, i=0)
    e.t += randexp(e.rate)
    if e.t > e.max
        return nothing
    end
    e.t, i + 1
end

randexp(rate) = - log(1 - rand()) / abs(rate)


# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct PoissonInpPopulation{T<:AbstractFloat} <: InputPopulation
    id::Int
    maxrate::T
    minrate::T
    sampleperiod::T
    spiketype::UnionAll
    out_spikes::Array{Spike, 1}
    length::Int

    function PoissonInpPopulation(id, sz, spiketype, maxrate=maxrate, minrate=minrate, sampleperiod=sampleperiod)
        (id < 1 || sz < 1) ? error("id and sz must be > 0") : nothing

        new{typeof(maxrate)}(id, maxrate, minrate, sampleperiod, spiketype, spiketype[], sz)
    end
end

function generate_input(pop::PoissonInpPopulation{T}, neuron_id::Int, sensor_inp::T, maxval::T) where T<:AbstractFloat
    rate = compute_rate(pop, maxval, sensor_inp)
    expacc = ExpAccumulate(rate, pop.sampleperiod)
    spikes = pop.spiketype[]
    for t in expacc
        push!(spikes, pop.spiketype(neuron_id, t))
    end
    spikes
end

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

randexp(rate) = - log(1 - rand()) / rate


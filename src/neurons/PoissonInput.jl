# Rates are in Hz
maxrate = 20.
minrate = 0.
period = 1000. # Milliseconds

struct PoissonInpPopulation{T<:AbstractFloat, S<:Spike} <: InputPopulation
    id::Int
    maxrate::T
    minrate::T
    period::T
    spiketype::UnionAll
    out_spikes::Queue{S}
    length::Int

    function PoissonInpPopulation(id, sz, spiketype, maxrate=maxrate, minrate=minrate, period=period)
        (id < 1 || sz < 1) ? error("id and sz must be > 0") : nothing

        out_spikes = Queue(spiketype)
        new{typeof(maxrate), spiketype}(id, maxrate, minrate, sampleperiod, spiketype, out_spikes, sz)
    end
end

function generate_input(pop::PoissonInpPopulation{T}, neuron_id::Int, sensor_inp::Vector{T}, maxval::T) where T<:AbstractFloat
    rate = compute_rate.(pop, maxval, sensor_inp)
    sampleperiod = pop.period / length(sensor_inp)
    expacc = ExpAccumulate.(rate, sampleperiod)
    spikes = pop.spiketype[]
    start = 0.
    for acc in expacc
        for t in acc
            push!(spikes, pop.spiketype(pop.id, neuron_id, t + start))
        end
        start += sampleperiod
    end
    spikes
end

reset!(pop::PoissonInpPopulation) = clear!(pop.out_spikes)

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


# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct PoissonInpNeuron{T<:AbstractFloat} <: InputNeuron
    id::UInt
    maxrate::T
    minrate::T
    sampleperiod::T

    function PoissonInpNeuron(id)
        PoissonInpNeuron(id, maxrate, minrate, sampleperiod)
    end

    function PoissonInpNeuron(id, maxrate, minrate, sampleperiod)
        new{typeof(maxrate)}(UInt(id), maxrate, minrate, sampleperiod)
    end
end

function generate_input(neuron::PoissonInpNeuron{T}, sensor_inp::T, maxval::T, spiketype) where T<:AbstractFloat
    rate = compute_rate(neuron, maxval, sensor_inp)
    expacc = ExpAccumulate(rate, neuron.sampleperiod)
    spikes = spiketype[]
    for t in expacc
        push!(spikes, spiketype(neuron.id, t))
    end
    spikes
end

compute_rate(neuron, maxval, val) = (val / maxval * (neuron.maxrate - neuron.minrate) + neuron.minrate) / 1000

randexp(rate) = - log(1 - rand()) / rate

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


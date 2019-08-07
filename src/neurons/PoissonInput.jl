# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct PoissonInpNeuron{T<:AbstractFloat} <: InputNeuron
    id::UInt
    maxrate::T
    minrate::T
    sampleperiod::T

    function PoissonInpNeuron(id::Int)
        PoissonInpNeuron(id, maxrate, minrate, sampleperiod)
    end

    function PoissonInpNeuron(id::Int, maxrate, minrate, sampleperiod)
        new{typeof(maxrate)}(UInt(id), maxrate, minrate, sampleperiod)
    end
end

function generate_input(neuron::PoissonInpNeuron{T}, maxval::T, sensor_inp::T, spiketype) where T<:AbstractFloat
    time = 0
    spikes = spiketype[]
    rate = compute_rate(maxval, sensor_inp)
    while (time < neuron.sampleperiod)
        dt = exprand(rate)
        time += dt
        push!(spikes, spiketype(neuron.id, dt))
    end
    spikes
end

compute_rate(maxval, val) = val / maxval * (maxrate - minrate) + minrate

exprand(rate) = - log(1 - rand()) / rate

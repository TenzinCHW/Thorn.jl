# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct RateInpNeuron{T<:AbstractFloat} <: InputNeuron
    maxrate::T
    minrate::T
    sampleperiod::T
end

function generate_input(neuron::RateInpNeuron{T}, sensor_inp::T, maxval::T, spiketype) where T<:AbstractFloat
    timespacing = 1 / compute_rate(maxval, sensor_inp)
    numiter = Int(floor(neuron.sampleperiod / timespacing))
    spikes = [spiketype(neuron.id, t * timespacing) for t in 1:numiter]
end

compute_rate(maxval, val) = (val / maxval * (maxrate - minrate) + minrate) / 1000


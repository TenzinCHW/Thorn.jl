# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct RateInpNeuron{T<:AbstractFloat} <: InputNeuron
    maxrate::T
    minrate::T
    sampleperiod::T
end

function generate_input(neuron::RateInpNeuron{T}, sensor_inp::T) where T<:AbstractFloat
end


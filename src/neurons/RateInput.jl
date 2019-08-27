# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct RateInpPopulation{T<:AbstractFloat} <: InputPopulation
    maxrate::T
    minrate::T
    sampleperiod::T
end

function generate_input(pop::RateInpPopulation, sensor_inp::T, maxval::T, spiketype) where T<:AbstractFloat
    timespacing = 1 / compute_rate(maxval, sensor_inp)
    numiter = Int(floor(pop.sampleperiod / timespacing))
    spikes = [spiketype(pop.id, t * timespacing) for t in 1:numiter]
end


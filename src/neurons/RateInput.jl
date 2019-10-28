# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 1000. # Milliseconds

struct RateInpPopulation{T<:AbstractFloat, S<:Spike} <: InputPopulation
    id::Int
    maxrate::T
    minrate::T
    sampleperiod::T
    spiketype::UnionAll
    out_spikes::Queue{S}
    length::Int

    function RateInpPopulation(id, sz, spiketype; maxrate=maxrate, minrate=minrate, sampleperiod=sampleperiod)
        (id < 1 || sz < 1) ? error("id and sz must be > 0") : nothing

        out_spikes = Queue{spiketype}
        new{typeof{maxrate}, spiketype}(id, maxrate, minrate, sampleperiod, spiketype, out_spikes, sz)
    end
end

function generate_input(pop::RateInpPopulation, neuron_id::Int, sensor_inp::Vector{T}, maxval::T) where T<:AbstractFloat
    timespacing = 1 ./ compute_rate.(maxval, sensor_inp)
    numiter = Int.(floor.(pop.sampleperiod ./ timespacing))
    spikes = pop.spiketype[]
    start = 0.
    for (interval, numiter) in zip(timespacing, numiters)
        for i in 1:numiter
            push!(spikes, pop.spiketype(pop.id, neuron_id, i * interval + start))
        end
        start += pop.sampleperiod
    end
    spikes
end

reset!(p::RateInpPopulation) = nothing


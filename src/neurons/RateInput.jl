# Rates are in Hz
maxrate = 20.
minrate = 1.
sampleperiod = 100. # Milliseconds

"""
    ```RateInpPopulation(
        id, sz spiketype;
        maxrate=20., minrate=0., sampleperiod=100., sign=1)```

Datastructure for generating spikes at a constant frequency.
Use this type as input to `Cortex`. See `Cortex`.
"""
struct RateInpPopulation{T<:AbstractFloat, S<:Spike} <: InputPopulation
    id::Int
    maxrate::T
    minrate::T
    sampleperiod::T
    spiketype::UnionAll
    out_spikes::Queue{S}
    length::Int
    sign::Int8

    function RateInpPopulation(
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

function generate_spikes!(
        pop::RateInpPopulation,
        sensor_inp::Array{T, 2},
        maxval::T) where T<:AbstractFloat
    for neuron_id in 1:pop.length
        data = sensor_inp[neuron_id, :]
        timespacing = 1 ./ compute_rate.(pop, maxval, data)
        numiters = Int.(floor.(pop.sampleperiod ./ timespacing))
        start = 0.
        for (interval, numiter) in zip(timespacing, numiters)
            if sign(interval) == pop.sign
                for i in 1:numiter
                    push!(pop.out_spikes, pop.spiketype(pop.id, neuron_id, i * abs(interval) + start, sign(interval)))
                end
            end
            start += pop.sampleperiod
        end
    end
end

reset!(p::RateInpPopulation) = nothing


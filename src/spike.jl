# Spikes are events; all their charge is delivered in an instant (like a Dirac-delta function).

abstract type Spike end

Base.length(s::Spike) = 1

Base.iterate(s::Spike) = s, nothing
Base.iterate(s::Spike, n::Nothing) = nothing

struct LIFSpike{T<:AbstractFloat} <: Spike
    pop_id::Int # Population index containing neuron that fired this spike
    neuron_id::Int # Localindex of neuron that fired this spike
    time::T # Global time in ms since start of simulation
    sign::Int8

    function LIFSpike(pop_id, neuron_id, time, sign)
        sign ∉ (-1, 1) && error("sign must be -1 or 1")
        new{typeof(time)}(pop_id, neuron_id, time, sign)
    end
end


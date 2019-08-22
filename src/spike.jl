# Spikes are events; all their charge is delivered in an instant (like a Dirac-delta function).

abstract type Spike end

Base.length(s::Spike) = 1

Base.iterate(s::Spike) = s, nothing
Base.iterate(s::Spike, n::Nothing) = nothing

struct LIFSpike{T<:AbstractFloat} <: Spike
    neuron_index::UInt # Localindex of neuron that fired this spike
    time::T # Global time in ms since start of simulation

    function LIFSpike(neuron_index::UInt, time::T) where T<:AbstractFloat
        if (neuron_index < 1)
            error("neuron_index must be and Int > 0")
        end
        if (time < 0)
            error("time must be <:AbstractFloat >= 0")
        end
        new{typeof(time)}(neuron_index, time)
    end
end


# Spikes are events; all their charge is delivered in an instant (like a Dirac-delta function).

abstract type Spike end

struct LIFSpike{T<:AbstractFloat}
    neuron_index::UInt # Localindex of neuron that fired this spike
    time::T # Global time in ms since start of simulation
end

struct SpikeQueue
    spikes::Array{Spike, 1}
end


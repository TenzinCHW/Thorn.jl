# Spikes are events; all their charge is delivered in an instant (like a Dirac-delta function).

struct Spike{T<:AbstractFloat}
    neuron_index::Int # Global index of neuron that fired this spike
    time::T # Global time in ms since start of simulation
end

struct SpikeQueue
    spikes::Array{Spike, 1}
end


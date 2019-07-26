include("neuron.jl")
include("../spike.jl")

mutable struct LIFNeuron{T<:AbstractFloat} <: ProcessingNeuron
    id::UInt
    u::T
    rest_u::T
    reset_u::T
    tau::T
    thresh::T
    resist::T
    last_out::Spike

    function LIFNeuron(id::UInt)
        # TODO Find good default values for reset_u, tau, thresh and resist
        new(id, 0, 0, -5, 1, ?, 0.5, nothing)
    end
end

function neuron_state_update(neuron::LIFNeuron, weight::AbstractFloat, spike::Spike, prev_spike::Spike)
    decayed = e ^ - (spike.time - prev_spike.time)
    if (neuron.u >= neuron.rest_u)
        decayed = - decayed
    end
    neuron.u = weight + decayed
end

function output_spike(neuron::LIFNeuron, spike::Spike, next_spike::Spike)
    if (neuron.u >= neuron.thresh && spike.time <= next_spike.time)
        neuron.u = neuron.reset_u
        neuron.last_out = Spike(neuron.id, spike.time)
        neuron.last_out
    end
end


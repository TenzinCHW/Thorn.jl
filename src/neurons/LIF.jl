mutable struct LIFNeuron{T<:AbstractFloat} <: ProcessingNeuron
    id::UInt
    u::T
    rest_u::T
    reset_u::T
    alpha::T
    tau::T
    thresh::T
    resist::T
    last_out::Union{Spike, Nothing}

    function LIFNeuron(id::Int, u::T, rest_u::T, reset_u::T, alpha::T, tau::T, thresh::T, resist::T) where T<:AbstractFloat
        new{typeof(u)}(UInt(id), u, rest_u, reset_u, alpha, tau, thresh, resist, nothing)
    end

    LIFNeuron(id::Int) = LIFNeuron(id, def_u, def_rest_u, def_reset_u, def_alpha, def_tau, def_thresh, def_resist)
end

# TODO Find good default values for reset_u, tau, thresh and resist
def_u = 0.
def_rest_u = 0.
def_reset_u = -5.
def_alpha = 1.
def_tau = 1.
def_thresh = 5.
def_resist = 0.5

function state_update!(neuron::LIFNeuron, weight::AbstractFloat, spike::S, prev_spike::Union{S, Nothing}) where S<:Spike
    prev_spike_t = (prev_spike == nothing) ? 0. : prev_spike.time
    c = - neuron.tau * log(abs(neuron.u - neuron.rest_u))
    dt = spike.time - prev_spike_t
    decayed = neuron.rest_u + flipsign(exp(-(dt + c) / neuron.tau), neuron.u)
    neuron.u = weight + decayed
end

function output_spike!(neuron::LIFNeuron, spike::T, next_spike::Union{T, Nothing}) where T<:Spike
    if (neuron.u >= neuron.thresh && (next_spike == nothing || spike.time <= next_spike.time))
        neuron.u = neuron.reset_u
        neuron.last_out = LIFSpike(neuron.id, spike.time)
        neuron.last_out
    end
end

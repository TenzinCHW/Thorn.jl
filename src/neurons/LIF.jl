mutable struct LIFNeuron{T<:AbstractFloat} <: ProcessingNeuron
    id::UInt
    u::T
    rest_u::T
    spike_u::T
    alpha::T
    tau::T
    thresh::T
    last_out::Union{Spike, Nothing}
    u_func::Function # u as a function of time diff between two spikes

    function LIFNeuron(id::Int, u::T, rest_u::T, reset_u::T, alpha::T, tau::T, thresh::T, u_func) where T<:AbstractFloat
        new{typeof(u)}(UInt(id), u, rest_u, reset_u, alpha, tau, thresh, nothing, u_func)
    end

    LIFNeuron(id::Int) = LIFNeuron(id, def_u, def_rest_u, def_spike_u, def_alpha, def_tau, def_thresh, dt->def_u)
end

# TODO Find good default values for reset_u, tau and thresh
def_u = 0.
def_rest_u = 0.
def_spike_u = -5.
def_alpha = 1.
def_tau = 1.
def_thresh = 5.

function state_update!(neuron::LIFNeuron, weight::AbstractFloat, spike::S, prev_spike::Union{S, Nothing}) where S<:Spike
    prev_spike_t = isnothing(prev_spike) ? 0. : prev_spike.time
    inter_spike_t = spike.time - prev_spike_t
    neuron.u_func = make_u_func(neuron.u, neuron.rest_u, neuron.tau, weight, inter_spike_t)
    neuron.u = neuron.u_func(inter_spike_t)
end

function output_spike!(neuron::LIFNeuron, spike::T, next_spike::Union{T, Nothing}) where T<:Spike
    if (neuron.u >= neuron.thresh && (isnothing(next_spike) || spike.time <= next_spike.time))
        neuron.u = neuron.spike_u
        neuron.last_out = LIFSpike(neuron.id, spike.time)
    end
end

function reset!(neuron::LIFNeuron)
    neuron.u = neuron.rest_u
end

function make_u_func(u, rest_u, tau, weight, inter_spike_t)
    function u_func(dt)
        c = - tau * log(abs(u - rest_u))
        decayed = rest_u + flipsign(exp(-(dt + c) / tau), u)
        dt >= inter_spike_t ? weight + decayed : decayed
    end
end


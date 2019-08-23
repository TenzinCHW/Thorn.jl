mutable struct LIFNeuron{T<:AbstractFloat} <: ProcessingNeuron
    id::UInt
    u::T
    rest_u::T
    spike_u::T
    alpha::T
    tau::T
    thresh::T
    homeostasis_a::T
    homeostasis_b::T
    last_out::Union{Spike, Nothing}
    u_func::Function # u as a function of time diff between two spikes

    function LIFNeuron(id::Int, u=def_u, rest_u=def_rest_u, spike_u=def_spike_u, alpha=def_alpha, tau=def_tau, thresh=def_thresh, homeostasis_a=def_homeostasis_a, homeostasis_b=def_homeostasis_b)
        u_func = make_u_func(rest_u, tau)
        new{typeof(u)}(UInt(id), u, rest_u, spike_u, alpha, tau, thresh, homeostasis_a, homeostasis_b, nothing, u_func)
    end

    #LIFNeuron(id::Int) = LIFNeuron(id, def_u, def_rest_u, def_spike_u, def_alpha, def_tau, def_thresh, dt->def_u)
end

function make_u_func(rest_u, tau)
    function u_func(u, dt)
        c = - tau * log(abs(u - rest_u))
        rest_u + flipsign(exp(-(dt + c) / tau), u)
    end
end

# TODO Find good default values for reset_u, tau and thresh
def_u = 0.
def_rest_u = 0.
def_spike_u = -5.
def_alpha = .8
def_tau = 16.8
def_thresh = 5.
def_homeostasis_a = 0.1
def_homeostasis_b = 0.01

function state_update!(neuron::LIFNeuron, weight::AbstractFloat, spike::S, prev_spike::Union{S, Nothing}) where S<:Spike
    prev_spike_t = isnothing(prev_spike) ? 0 : prev_spike.time
    inter_spike_t = spike.time - prev_spike_t
    #c = - neuron.tau * log(abs(neuron.u - neuron.rest_u))
    #decayed = neuron.rest_u + flipsign(exp(-(inter_spike_t + c) / neuron.tau), neuron.u)
    #neuron.u = weight + decayed
    neuron.u = weight + neuron.u_func(neuron.u, inter_spike_t)
end

function output_spike!(neuron::LIFNeuron, spike::T, next_spike::Union{T, Nothing}) where T<:Spike
    if (neuron.u >= neuron.thresh && (isnothing(next_spike) || spike.time <= next_spike.time))
        neuron.u = neuron.spike_u
        neuron.thresh += exp(neuron.homeostasis_a * neuron.thresh + neuron.homeostasis_b)
        neuron.last_out = LIFSpike(neuron.id, spike.time)
    end
end

function reset!(neuron::LIFNeuron)
    neuron.u = neuron.rest_u
end


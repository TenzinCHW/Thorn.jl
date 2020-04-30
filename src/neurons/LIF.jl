# This is an implementation of the exponential leaky integrate and fire model
def_u = 0.
def_rest_u = 0.
def_spike_u = -5.
def_α = .8
def_τ = 16.8
def_γ1 = 0.01
def_γ2 = 0.001
def_thresh = 1.
def_arp = 2.
def_rrp = 20.

struct LIFPopulation{T<:AbstractFloat} <: ProcessingPopulation
    id::Int
    length::Int
    u::Vector{T}
    init_u::T
    rest_u::T
    spike_u::T
    α::T
    τ::T
    γ1::T
    γ2::T
    arp::T
    rrp::T
    fire_after::Vector{T}
    thresh::Vector{T}
    num_spikes::Vector{UInt}
    u_func::Function
    weight_update::Function
    η::T # Learning rate
    out_spikes::Queue{LIFSpike}
    last_spike::Array{Union{Spike, Nothing}, 0}

    function LIFPopulation(id, sz, weight_update, η; init_u=def_u,
                           rest_u=def_rest_u, spike_u=def_spike_u,
                           α=def_α, τ=def_τ, γ1=def_γ1, γ2=def_γ2,
                           threshval=def_thresh, arp=def_arp,
                           rrp=def_rrp
                          )
        (id < 1 || sz < 1) ? error("sz and id must be > 0") : nothing
        !isa(weight_update, Function) ? error("weight_update must be a function") : nothing
        u = init_u * ones(sz)
        fire_after = zeros(sz)
        thresh = threshval * ones(sz)
        num_spikes = zeros(UInt, sz)
        u_func = LIF(rest_u, τ)
        last_spike = Array{Union{Spike, Nothing}}(undef)
        last_spike[] = nothing
        q = Queue(LIFSpike) #Queue(typeof(LIFSpike(1, 1, arp)))
        new{typeof(η)}(
            id,
            sz,
            u,
            init_u,
            rest_u,
            spike_u,
            α,
            τ,
            γ1,
            γ2,
            arp,
            rrp,
            fire_after,
            thresh,
            num_spikes,
            u_func,
            weight_update,
            η,
            q,
            last_spike)
    end
end

function recvspike!(
        pop::LIFPopulation,
        proposedspikes::Vector{Spike},
        weights::Array{T, 2},
        spike::Spike) where T<:AbstractFloat
    weights = weights[:, spike.neuron_id]
    dt = isnothing(pop.last_spike[]) ? spike.time : spike.time - pop.last_spike[].time
    pop.u .= (spike.time .>= pop.fire_after) .* weights + pop.u_func.(pop.u, dt) * spike.sign
    pop.last_spike[] = spike
    fired = pop.u .>= pop.thresh
    inds = findall(fired)
    for i in inds
        fire_time = spike.time + pop.arp
        pop.fire_after[i] = spike.time + pop.rrp
        push!(proposedspikes, LIFSpike(pop.id, i, fire_time, 1))#sign(pop.u[i]))) # Last arg is sign
    end
end

function LIF(rest_u, τ)
    function u_func(u, δt)
        rest_u + (u - rest_u) * exp(-δt / τ)
    end
end

function updatevalidspikes!(pop::LIFPopulation, spikes::Vector{S}) where S<:Spike
    if !isempty(spikes)
        pop.u .= pop.spike_u # WTA circuit
        update_spike!.(pop, spikes)
    end
end

function update_spike!(pop::LIFPopulation, s::Spike)
    #pop.thresh[s.neuron_id] += pop.γ1 * exp(pop.γ2*length(pop.num_spikes[s.neuron_id]))
    #pop.num_spikes[s.neuron_id] += 1
    pop.thresh[s.neuron_id] = (1 + pop.γ1) * pop.u[s.neuron_id]
    pop.u[s.neuron_id] = pop.rest_u
end

function reset!(pop::LIFPopulation)
    pop.last_spike[] = nothing
    pop.fire_after .= 0
    pop.u .= pop.init_u
end


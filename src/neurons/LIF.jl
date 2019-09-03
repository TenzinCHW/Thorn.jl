# This is an implementation of the exponential leaky integrate and fire model
def_u = 0.
def_rest_u = 0.
def_spike_u = -5.
def_α= .8
def_τ = 16.8
def_thresh = 2.
def_arp = 20.

struct LIFPopulation{T<:AbstractFloat} <: ProcessingPopulation
    id::Int
    length::Int
    u::Array{T, 1}
    init_u::T
    rest_u::T
    spike_u::T
    α::T
    τ::T
    arp::T
    thresh::Array{T, 1}
    u_func::Function
    weight_update::Function
    η::T # Learning rate
    out_spikes::Stack{LIFSpike}
    last_spike::Array{Union{Spike, Nothing}, 0}

    function LIFPopulation(id, sz, weight_update, η; init_u=def_u,
                           rest_u=def_rest_u, spike_u=def_spike_u,
                           α=def_α, τ=def_τ, threshval=def_thresh,
                           arp=def_arp
                          )
        (id < 1 || sz < 1) ? error("sz and id must be > 0") : nothing
        !isa(weight_update, Function) ? error("weight_update must be a function") : nothing
        u = init_u * ones(sz)
        thresh = threshval * ones(sz)
        u_func = LIF(rest_u, τ)
        last_spike = Array{Union{Spike, Nothing}}(undef)
        last_spike[] = nothing
        new{typeof(η)}(id, sz, u, init_u, rest_u, spike_u, α, τ,
                       arp, thresh, u_func, weight_update, η,
                       Stack(LIFSpike), last_spike
                       )
    end
end

function update_state!(pop::LIFPopulation, weights::SubArray{T, 1}, spike::Spike) where T<:AbstractFloat
    dt = isnothing(pop.last_spike[]) ? spike.time : spike.time - pop.last_spike[].time
    pop.u .= weights + pop.u_func.(pop.u, dt)
    pop.last_spike[] = spike
end

function LIF(rest_u, τ)
    function u_func(u, δt)
        c = - τ * log(abs(u - rest_u))
        rest_u + flipsign(exp(-(δt + c) / τ), u)
    end
end

function output_spike!(pop::LIFPopulation, spike::Spike)
    fired = pop.u .>= pop.thresh
    inds = findall(fired)
    [LIFSpike(pop.id, i, spike.time + pop.arp) for i in inds]
end

function update_spike!(pop::LIFPopulation, spikes::Vector{Spike})
    for s in spikes
        pop.u[s.neuron_id] = pop.u_rest
    end
end

function reset!(pop::LIFPopulation)
    pop.last_spike[] = nothing
    pop.u .= pop.init_u
end


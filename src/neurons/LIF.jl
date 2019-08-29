# This is an implementation of the exponential leaky integrate and fire model
def_u = 0.
def_rest_u = 0.
def_spike_u = -5.
def_α= .8
def_τ = 16.8
def_thresh = 2.

struct LIFPopulation{T<:AbstractFloat} <: ProcessingPopulation
    id::Int
    u::Array{T, 1}
    init_u::T
    rest_u::T
    spike_u::T
    α::T
    τ::T
    thresh::Array{T, 1}
    last_out::Array{Union{Spike, Nothing}, 1}
    length::Int
    u_func::Function
    weight_update::Function
    η::T # Learning rate
    out_spikes::Array{Spike, 1}
    last_spike::Union{Spike, Nothing}

    function LIFPopulation(id, sz, weight_update, η, init_u=def_u,
                           rest_u=def_rest_u, spike_u=def_spike_u,
                           α=def_α, τ=def_τ, threshval=def_thresh)
        (id < 1 || sz < 1) ? error("sz and id must be > 0") : nothing
        !isa(weight_update, Function) ? error("weight_update must be a function") : nothing
        u = init_u * ones(sz)
        thresh = threshval * ones(sz)
        last_out = [nothing for _ in 1:sz]
        u_func = LIF(rest_u, τ)
        new{typeof(η)}(id, u, init_u, rest_u, spike_u, α, τ, thresh, last_out,
                        sz, u_func, weight_update, η, Spike[], nothing
                       )
    end
end

function process_spike!(pop::LIFPopulation, weights::SubArray{T, 1}, spike::Spike) where T<:AbstractFloat
    dt = isnothing(pop.last_spike) ? spike.time : spike.time - pop.last_spike.time
    pop.u .= weights + pop.u_func.(pop.u, dt)
end

function LIF(rest_u, τ)
    function u_func(u, δt)
        c = - τ * log(abs(u - rest_u))
        rest_u + flipsign(exp(-(δt + c) / τ), u)
    end
end

function output_spike!(pop::LIFPopulation, spike::Spike, next_spike::Union{Spike, Nothing})
    if isnothing(next_spike) || spike.time <= next_spike.time
        fired = pop.u .>= pop.thresh
        inds = findall(fired)
        for i in inds
            pop.last_out[i] = LIFSpike(i, spike.time)
        end
        # filter this based on the timing of next_spike coming into the pop then sort the filtered array
        insertsorted(pop.out_spikes, Array{Spike, 1}(pop.last_out[inds]))
    end
end

function reset!(pop::LIFPopulation)
    pop.u .= pop.init_u
end


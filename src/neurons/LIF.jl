def_u = 0.
def_rest_u = 0.
def_spike_u = -5.
def_alpha = .8
def_tau = 16.8
def_thresh = 5.
def_homeostasis_a = 0.1
def_homeostasis_b = 0.01

struct LIFPopulation{T<:AbstractFloat} <: ProcessingPopulation
    id::Int
    u::Array{T, 1}
    rest_u::T
    spike_u::T
    alpha::T
    tau::T
    thresh::Array{T, 1}
    homeostasis_a::T
    homeostasis_b::T
    last_out::Array{Union{Spike, Nothing}, 1}
    length::Int
    u_func::Function
    weight_update::Function
    lr::T # Learning rate
    out_spikes::Array{Spike, 1}
    last_spike::Union{Spike, Nothing}

    function LIFPopulation(id, sz, weight_update, lr, u_val=def_u, rest_u=def_rest_u, spike_u=def_spike_u, alpha=def_alpha, tau=def_tau, threshval=def_thresh, homeostasis_a=def_homeostasis_a, homeostasis_b=def_homeostasis_b)
        (id < 1 || sz < 1) ? error("sz and id must be > 0") : nothing
        !isa(weight_update, Function) ? error("weight_update must be a function") : nothing
        u = u_val * ones(sz)
        thresh = threshval * ones(sz)
        last_out = [nothing for _ in 1:sz]
        u_func = LIF(rest_u, tau)
        new{typeof(lr)}(id, u, rest_u, spike_u, alpha, tau,
                        thresh, homeostasis_a, homeostasis_b,
                        last_out, sz, u_func, weight_update, lr,
                        Spike[], nothing
                       )
    end
end

function process_spike!(pop::LIFPopulation, weights::Array{T, 1}, spike::Spike) where T<:AbstractFloat
    dt = isnothing(pop.last_spike) ? spike.time : spike.time - pop.last_spike.time
    for i in pop.length
        pop.u[i] = weights[i] + pop.u_func(pop.u[i], dt)
    end
end

function u_func(u, dt)
    LIF(0., 16.8)(u, dt)
end

function LIF(rest_u, tau)
    function u_func(u, dt)
        c = - tau * log(abs(u - rest_u))
        rest_u + flipsign(exp(-(dt + c) / tau), u)
    end
end

function output_spike!(pop::LIFPopulation, spike::Spike, next_spike::Union{Spike, Nothing})
    if isnothing(next_spike) || spike.time <= next_spike.time
        inds = findall(x->x>0, pop.u)
        for i in inds
            pop.thresh[i] += exp(pop.homeostasis_a * pop.thresh[i] + pop.homeostasis_b)
            pop.last_out[i] = LIFSpike(i, spike.time)
        end
        # filter this based on the timing of next_spike coming into the pop then sort the filtered array
        insertsorted(pop.out_spikes, Array{Spike, 1}(pop.last_out[inds]))
    end
end

function reset!(pop::LIFPopulation)
    pop.u .= pop.rest_u
end


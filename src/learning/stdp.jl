struct STDPWeights<:Weights
    value
    weight_update::Function
    αp::AbstractFloat
    αn::AbstractFloat
    τp::AbstractFloat
    τn::AbstractFloat
    lr::AbstractFloat

    function STDPWeights(
            value; weight_update=stdp!, αp=.8, αn=.5, τp=5., τn=3., lr=.1)
        new(value, weight_update, αp, αn, τp, τn, lr)
    end
end

# This function assumes that weights.value is a 2D array
"""
    stdp!(weights::Weights, pre::S, post::S) where {T<:AbstractFloat, S<:Spike}
STDP weight update function. Pass this as a connection parameter. See `Cortex`.
"""
function stdp!(weights::STDPWeights, pre::S, post::S) where {S<:Spike}
    weightval = weights.value
    αp, αn, τp, τn = weights.αp, weights.αn, weights.τp, weights.τn
    if pre.time < post.time
        update = αp * exponent(pre.time, post.time, τp)
    else
        update = -αn * exponent(-pre.time, -post.time, τn)
    end
    w = weightval[post.neuron_id, pre.neuron_id]
    weights.value[post.neuron_id, pre.neuron_id] = w + weights.lr * update
end

exponent(pre::T, post::T, τ::T) where T<:AbstractFloat = exp((pre - post) / τ)


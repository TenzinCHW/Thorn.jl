"""
    `STDPWeights(value; αp=.8, αn=.5, τp=5., τn=3., lr=.1)`
Constructs a datastructure for storing weight values and related parameters for updating the weights.
`value` is the initial value of the weights.
`αp` is α_{+} while `αn` is α_{-}.
`τp` is τ_{+} while `τn` is τ_{-}.
"""
struct STDPWeights{T<:AbstractFloat}<:Weights
    value::Array{T, 2}
    αp::AbstractFloat
    αn::AbstractFloat
    τp::AbstractFloat
    τn::AbstractFloat
    lr::AbstractFloat

    function STDPWeights(
            value::Array{T, 2};
            αp=.8, αn=.5, τp=5., τn=3., lr=.1) where T<:AbstractFloat
        new{T}(value, αp, αn, τp, τn, lr)
    end
end

function updateweights!(weights::STDPWeights, pre, post)
    updateweights!.(weights, pre, post)
end

# This function assumes that weights.value is a 2D array
"""
    ```updateweights!(
        weights::STDPWeights, pre::S, post::S) where {T<:AbstractFloat, S<:Spike}```
STDP weight update function. Pass this as a connection parameter. See `Cortex`.

If pre.time < post.time,
Δw = α_{+} * e^{\frac{pre.time - post.time}{\tau_{+}}}
else,
Δw = -α_{-} * e^{\frac{post.time - pre.time}{\tau_{-}}}
"""
function updateweights!(weights::STDPWeights, pre::S, post::S) where {S<:Spike}
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


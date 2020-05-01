# This function assumes that the weights are a 2D array
function stdp!(
        weights::Weights,
        pop::NeuronPopulation,
        pre::S,
        post::S) where {T<:AbstractFloat, S<:Spike}
    weightval = weights.value
    α, τ = pop.α, pop.τ
    if pre.time < post.time
        update = α * exponent(pre.time, post.time, τ)
    else
        update = -exponent(-pre.time, -post.time, τ)
    end
    w = weightval[post.neuron_id, pre.neuron_id]
    weights.value[post.neuron_id, pre.neuron_id] = w + weights.lr * update
end

exponent(pre::T, post::T, τ::T) where T<:AbstractFloat = exp((pre - post) / τ)


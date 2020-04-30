# Okay this part is a bit more confusing...
# Should I apply stdp for every pair of input and output spikes or just the last input and last output spike?
# If I do just last input and last output, I just have to apply a weight update before processing each input spike and after each output spike
function stdp!(
        weights::Array{T, 2},
        pop::NeuronPopulation,
        pre::S,
        post::S) where {T<:AbstractFloat, S<:Spike}
    α, η, τ = pop.α, pop.η, pop.τ
    if pre.time < post.time
        update = pop.α * exponent(pre.time, post.time, pop.τ)
    else
        update = -exponent(-pre.time, -post.time, pop.τ)
    end
    w = weights[post.neuron_id, pre.neuron_id]
    weights[post.neuron_id, pre.neuron_id] = w + pop.η * update
end

exponent(pre::T, post::T, τ::T) where T<:AbstractFloat = exp((pre - post) / τ)


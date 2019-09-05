# Okay this part is a bit more confusing...
# Should I apply stdp for every pair of input and output spikes or just the last input and last output spike?
# If I do just last input and last output, I just have to apply a weight update before processing each input spike and after each output spike
function stdp(pop::NeuronPopulation, w::AbstractFloat, pre::S, post::S) where S<:Spike
    α, η, τ = pop.α, pop.η, pop.τ
    if pre.time < post.time
        update = pop.α * exponent(pre.time, post.time, pop.τ)
    else
        update = exponent(-pre.time, -post.time, pop.τ)
    end
    w + pop.η * update
end

exponent(pre::T, post::T, τ::T) where T<:AbstractFloat = exp((pre - post) / τ)


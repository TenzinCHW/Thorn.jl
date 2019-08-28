# Okay this part is a bit more confusing...
# Should I apply stdp for every pair of input and output spikes or just the last input and last output spike?
# If I do just last input and last output, I just have to apply a weight update before processing each input spike and after each output spike
function stdp(pop::NeuronPopulation, w::T, last_out::Union{Spike, Nothing}, prev_spike::Spike, next_spike::Union{Spike, Nothing}) where T<:AbstractFloat
    alpha, lr, tau = pop.alpha, pop.lr, pop.tau
    if !isnothing(last_out) && !isnothing(next_spike)
        # Update based on time difference between prev_spike and neuron.last_out
        weaken = - alpha * exponent(last_out.time, next_spike.time, tau)
        # Update based on time difference between neuron.last_out and next_spike
        strengthen = exponent(-prev_spike.time, -last_out.time, tau)
        return lr * (strengthen + weaken)
    end
    w
end

exponent(pre::T, post::T, tau::T) where T<:AbstractFloat = exp((pre - post) / tau)


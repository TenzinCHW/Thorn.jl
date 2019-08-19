# Okay this part is a bit more confusing...
# Should I apply stdp for every pair of input and output spikes or just the last input and last output spike?
# If I do just last input and last output, I just have to apply a weight update before processing each input spike and after each output spike
function stdp(neuron::Neuron, lr::T, w::T, prev_spike::Spike, next_spike::Union{Spike, Nothing}) where T<:AbstractFloat
    alpha, last_out, tau = neuron.alpha, neuron.last_out, neuron.tau
    if !any(isnothing.([last_out, next_spike]))
        # Update based on time difference between prev_spike and neuron.last_out
        weaken = - alpha * exponent(last_out.time, next_spike.time, tau)
        # Update based on time difference between neuron.last_out and next_spike
        strengthen = exponent(prev_spike.time, last_out.time, tau)
        return lr * (strengthen + weaken)
    end
    w
end

function exponent(pre, post, tau)
    exp((post - pre) / tau)
end


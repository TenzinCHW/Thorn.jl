# Okay this part is a bit more confusing...
# Should I apply stdp for every pair of input and output spikes or just the last input and last output spike?
# If I do just last input and last output, I just have to apply a weight update before processing each input spike and after each output spike
function stdp!(neuron::Neuron, w::AbstractFloat, prev_spike::Spike, next_spike::Spike)
    # TODO Update based on time difference between prev_spike and neuron.last_out
    # TODO Update based on time difference between neuron.last_out and next_spike
end


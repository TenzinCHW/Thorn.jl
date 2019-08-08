module SpikingNN
    include("spike.jl")
    include("neuron.jl")
    include("neurons/LIF.jl")
    include("neurons/PoissonInput.jl")
    include("structure/population.jl")
    include("structure/cortex.jl")
    include("learning/stdp.jl")
    include("newtypes.jl")

    export Spike, LIFSpike,
    Neuron, ProcessingNeuron, InputNeuron,
    LIFNeuron, PoissonInpNeuron, RateInpNeuron,
    state_update!, output_spike!, generate_input,
    NeuronPopulation, InputNeuronPopulation, ProcessingNeuronPopulation,
    process_spike!, generate_input_spikes!,
    stdp,
    Cortex,
    process_sample!
end

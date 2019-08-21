module Thorn
    include("spike.jl")
    include("neuron.jl")
    include("neurons/LIF.jl")
    include("neurons/PoissonInput.jl")
    include("structure/population.jl")
    include("structure/cortex.jl")
    include("learning/stdp.jl")
    include("newtypes.jl")
    include("monitor/monitor.jl")
    include("monitor/plotting.jl")
    include("data/persistence.jl")
    include("data/data.jl")

    export Spike, LIFSpike,
    Neuron, ProcessingNeuron, InputNeuron,
    LIFNeuron, PoissonInpNeuron, RateInpNeuron,
    state_update!, output_spike!, reset!, generate_input,
    NeuronPopulation, InputNeuronPopulation, ProcessingNeuronPopulation,
    process_spike!, generate_input_spikes!,
    stdp,
    Cortex,
    process_sample!, population_dependency, dependent_populations,
    Monitor,
    monitor!, collapse, rasterspikes, gridify,
    Datasource,
    datasrcreadbits, datasrcwritebits!, datasrcread, datasrcwrite!, datasrcitems,
    Dataset,
    swaptraintest!, resizeset!, shufflebyclass!,
    Dataloader
end

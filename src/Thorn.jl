module Thorn
    include("spike.jl")
    include("customdatastructs.jl")
    include("structure/weights.jl")
    include("structure/population.jl")
    include("structure/cortex.jl")
    include("neurons/LIF.jl")
    include("neurons/Input.jl")
    include("neurons/PoissonInput.jl")
    include("neurons/RateInput.jl")
    include("learning/stdp.jl")
    include("newtypes.jl")
    include("monitor/monitor.jl")
    include("monitor/plotting.jl")
    include("data/persistence.jl")
    include("data/data.jl")
    include("train/classify.jl")

    export Spike, LIFSpike,
    Queue, insertsorted!,
    NeuronPopulation, InputPopulation, ProcessingPopulation, InputPopulationPair,
    process_spike!, generate_input_spikes!, insertsorted,
    LIFPopulation, PoissonInpPopulation, RateInpPopulation,
    recvspike!, updatevalidspikes!, reset!, compute_rate, generate_input,
    stdp!,
    Cortex,
    process_sample!, population_dependency, dependent_populations, freeze_weights!, unfreeze_weights!,
    Monitor,
    getspike!, spikesfromrecord, getweights!, getu!, getvarfrompop, monitor!, collapserecord!, rasterspikes, gridify,
    Datasource,
    datasrcreadbits, datasrcwritebits!, datasrcread, datasrcwrite!, datasrcitems,
    Dataset,
    swappartition!, resizeprop!, shufflebyclass!,
    Dataloader
end

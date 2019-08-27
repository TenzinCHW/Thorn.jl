module Thorn
    include("spike.jl")
    include("structure/population.jl")
    include("structure/cortex.jl")
    include("neurons/LIF.jl")
    #include("neurons/SRM.jl")
    include("neurons/Input.jl")
    include("neurons/PoissonInput.jl")
    include("neurons/RateInput.jl")
    include("learning/stdp.jl")
    include("newtypes.jl")
    include("monitor/monitor.jl")
    include("monitor/plotting.jl")
    include("data/persistence.jl")
    include("data/data.jl")

    export Spike, LIFSpike,
    NeuronPopulation, InputPopulation, ProcessingPopulation,
    process_spike!, generate_input_spikes!, insertsorted,
    LIFPopulation, PoissonInpPopulation, RateInpPopulation,
    state_update!, output_spike!, reset!, compute_rate, generate_input,
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

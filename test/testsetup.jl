function defaultparams()
    sz = 5
    lfn = stdp
    lr = 0.1
    spiketype = LIFSpike
    return sz, lfn, lr, spiketype
end

function createprocpop(id::Int)
    LIFPopulation(id, defaultparams()[1:end-1]...) # Don't need spiketype for LIFPop
end

function createcortex(;inp_kwargs::Dict=Dict(), proc_kwargs::Dict=Dict(),
                     wt_init::Function=rand)
    sz, lfn, lr, spiketype = defaultparams()
    #input_neuron_types = [(PoissonInpPopulation, sz, inp_kwargs)]
    input_neuron_types = [InputPopulationPair(PoissonInpPopulation, sz, inp_kwargs)...]
    neuron_types = [(LIFPopulation, sz, lfn, lr, proc_kwargs)]
    conn = [1=>3, 2=>3]
    Cortex(input_neuron_types, neuron_types, conn, wt_init, spiketype)
end

function getrandomdata(cortex::Cortex, numsample::Int)
    sz = [pop.length for pop in cortex.input_populations]
    [i .- 1 for i in 2rand.(sz, numsample)]
end

function runprocesssample(args...)
    runprocesssample(args..., Dict()...)
end

function runprocesssample(args...; kwargs...)
    process_sample!(args...; kwargs...)
end


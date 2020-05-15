function defaultparams()
    sz = 5
    spiketype = LIFSpike
    lr = .15
    return sz, spiketype, lr
end

function createprocpop(id::Int)
    LIFPopulation(id, defaultparams()[1:end-1]...) # Don't need spiketype for LIFPop
end

function createcortex(;inp_kwargs::Dict=Dict(), proc_kwargs::Dict=Dict(),
                     wt_init::Function=rand)
    sz, spiketype, lr = defaultparams()
    #input_neuron_types = [(PoissonInpPopulation, sz, inp_kwargs)]
    input_neuron_types = [InputPopulationPair(PoissonInpPopulation, sz, inp_kwargs)...]
    neuron_types = [(LIFPopulation, sz, proc_kwargs)]
    #connparams = Dict(:lr=>lr)
    conn = [(1=>3, STDPWeights, wt_init),#, connparams),
            (2=>3, STDPWeights, wt_init)]#, connparams)]
    Cortex(input_neuron_types, neuron_types, conn, spiketype)
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


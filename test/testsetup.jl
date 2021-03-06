function defaultparams()
    sz = 5
    spiketype = LIFSpike
    lr = .15
    return sz, spiketype, lr
end

function createprocpop(id::Int)
    LIFPopulation(id, defaultparams()[1:end-1]...) # Don't need spiketype for LIFPop
end

function createcortex(;inp_kwargs::Dict=Dict(),
                      proc_kwargs::Dict=Dict(),
                      wt_init::Function=rand)
    sz, spiketype, lr = defaultparams()
    #input_neuron_types = [(PoissonInpPopulation, sz, inp_kwargs)]
    input_neuron_types = [InputPopulationPair(PoissonInpPopulation, sz, inp_kwargs)...]
    neuron_types = [(LIFPopulation, sz, proc_kwargs)]
    connparams = Dict(:lr=>lr, :wt_init=>wt_init)
    conn = [(1=>3, STDPWeights, connparams),
            (2=>3, STDPWeights, connparams)]
    Cortex(input_neuron_types, neuron_types, conn, spiketype)
end

function getrandomdata(cortex::Cortex, numsample::Int)
    sz = [pop.length for pop in cortex.input_populations]
    rng = Random.MersenneTwister(42)
    randstuff = [rand(rng, s, numsample) for s in sz]
    [i .- 1 for i in 2randstuff]
end

function runprocesssample(args...)
    runprocesssample(args..., Dict()...)
end

function runprocesssample(args...; kwargs...)
    process_sample!(args...; kwargs...)
end


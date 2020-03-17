function defaultparams()
    id = 1
    sz = 5
    lr = 0.1
    lfn = stdp
    spiketype = LIFSpike
    return id, sz, lr, lfn, spiketype
end

function createcortex(;inp_kwargs::Dict=Dict(), proc_kwargs::Dict=Dict(),
                     wt_init::Function=rand)
    _, sz, lr, lfn, spiketype = defaultparams()
    input_neuron_types = [(PoissonInpPopulation, sz, inp_kwargs)]
    neuron_types = [(LIFPopulation, sz, lfn, lr, proc_kwargs)]
    conn = [1=>2]
    Cortex(input_neuron_types, neuron_types, conn, wt_init, spiketype)
end


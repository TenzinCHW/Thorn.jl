inp_kwargs = Dict(:sampleperiod=>50.) # 50 ms sample_period
proc_kwargs = Dict(:τ=>20.4)
cortex = createcortex(;inp_kwargs=inp_kwargs, proc_kwargs=proc_kwargs)
numsample = 4
data = getrandomdata(cortex, numsample)
runprocesssample(cortex, data, 1.)
@test all(isa.([pop.out_spikes for pop in cortex.populations], Queue{spiketype}))
@test cortex.input_populations[1].sampleperiod == 50.
@test cortex.processing_populations[1].τ == 20.4


inp_kwargs = Dict(:sampleperiod=>50.) # 50 ms sample_period
proc_kwargs = Dict(:τ=>20.4)
wt_init() = 5 * rand()
wt_init(n, m) = 30 * rand(n, m)
cortex = createcortex(;inp_kwargs=inp_kwargs, proc_kwargs=proc_kwargs, wt_init=wt_init)
numsample = 4
data = getrandomdata(cortex, numsample)

function weightsonly(weights::Dict)
    [v.value for (k, v) in weights]
end

oldweights = deepcopy(cortex.weights)
runprocesssample(cortex, data, 1.)
@test all(isa.([pop.out_spikes for pop in cortex.populations], Queue{spiketype}))
@test cortex.input_populations[1].sampleperiod == 50.
@test cortex.processing_populations[1].τ == 20.4
@test weightsonly(oldweights) != weightsonly(cortex.weights)

oldweights = deepcopy(cortex.weights)
runprocesssample(cortex, data, 1.; train=false)
@test weightsonly(oldweights) == weightsonly(cortex.weights)

freeze_weights!(cortex, 1=>3)
runprocesssample(cortex, data, 1.)
@test weightsonly(oldweights)[1] == weightsonly(cortex.weights)[1]
# We did not freeze the 2=>3 layer of weights
@test weightsonly(oldweights)[2] != weightsonly(cortex.weights)[2]


sz = 5
numsample = 4
inp_kwargs = Dict(:sampleperiod=>50) # 50 ms sample_period
input_neuron_types = [(PoissonInpPopulation, sz, inp_kwargs)]
neuron_types = [(LIFPopulation, sz, stdp, 0.01)]
conn = [1=>2]
spiketype = LIFSpike
cortex = Cortex(input_neuron_types, neuron_types, conn, spiketype)
data = [rand(sz, numsample)]
extractors = Dict("nothing"=>(c, ps)->Dict())
spikes, _ = process_sample!(cortex, data, 1., extractors)
@test isa(spikes, Vector{Spike})


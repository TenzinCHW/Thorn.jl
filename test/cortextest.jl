sz = 5
input_neuron_types = [(PoissonInpPopulation, sz)]
neuron_types = [(LIFPopulation, sz, stdp, 0.01)]
conn = [1=>2]
spiketype = LIFSpike
cortex = Cortex(input_neuron_types, neuron_types, conn, spiketype)
data = [rand(sz)]
extractors = Dict("nothing"=>(c, ps)->Dict())
spikes, _ = process_sample!(cortex, data, 1., extractors)
@test isa(spikes, Array{Tuple{Int, Spike}, 1})


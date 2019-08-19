input_neuron_types = [(PoissonInpNeuron, 5)]
neuron_types = [(LIFNeuron, 5, stdp, 0.01)]
conn = Dict(1=>2)
spiketype = LIFSpike
cortex = Cortex(input_neuron_types, neuron_types, conn, spiketype)
data = [rand(sz)]
spikes, _ = process_sample!(cortex, data, 1.)
@test isa(spikes, Array{Tuple{UInt, Spike}, 1})


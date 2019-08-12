input_neuron_types = [(PoissonInpNeuron, 5)]
neuron_types = [(LIFNeuron, 5, stdp, 0.01)]
conn = BitArray([[0 0];
                 [1 0]])
spiketype = LIFSpike
cortex = Cortex(input_neuron_types, neuron_types, conn, spiketype)
data = [rand(sz)]
process_sample!(cortex, data, 1.)


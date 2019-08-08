# InputNeuron test
id = 1
neuron = PoissonInpNeuron(id)
data = rand()
maxval = 1.
spikes = generate_input(neuron, data, maxval, LIFSpike)
@test isa(spikes, Array{LIFSpike, 1})
@test length(spikes) > 0

# ProcessingNeuron test
id = 1
neuron = LIFNeuron(id)
inp_s1 = LIFSpike(UInt(1), 2.)
state_update!(neuron, 10., inp_s1, nothing)
@test neuron.u > 0.
inp_s2 = LIFSpike(UInt(1), 5.)
spike = output_spike!(neuron, inp_s1, inp_s2)
@test isa(spike, LIFSpike)


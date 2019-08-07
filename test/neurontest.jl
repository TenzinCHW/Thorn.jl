neuron = LIFNeuron(1)
inp_s1 = LIFSpike(UInt(1), 2.)
state_update!(neuron, 10., inp_s1, nothing)
@test neuron.u > 0.
inp_s2 = LIFSpike(UInt(1), 5.)
spike = output_spike!(neuron, inp_s1, inp_s2)
@test isa(spike, LIFSpike)


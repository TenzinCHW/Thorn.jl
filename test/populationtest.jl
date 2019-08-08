# Processing population test
id = 1
sz = 5
lr = 0.1
weights = 5 * ones(sz, sz)
spiketype = LIFSpike
inp_pop = InputNeuronPopulation(id, PoissonInpNeuron, sz, spiketype)
proc_pop = ProcessingNeuronPopulation(id, LIFNeuron, sz, stdp, lr)
data = ones(sz)
maxval = max(data...)
generate_input_spikes!(inp_pop, data, maxval)
for s in inp_pop.out_spikes
    process_spike!(proc_pop, weights[:, Int(s.neuron_index)], s)
end
voltages = [n.u for n in proc_pop.neurons]
@test all(voltages .!= 0)
@test length(proc_pop.out_spikes) == 0
output_spike!(proc_pop, inp_pop.out_spikes[end], inp_pop.out_spikes[1])
num_spike = sum(voltages .> proc_pop.neurons[1].thresh)
@test length(proc_pop.out_spikes) == num_spike

# Input population test
id = 1
sz = 5
spiketype = LIFSpike
inp_pop = InputNeuronPopulation(id, PoissonInpNeuron, sz, spiketype)
@test length(inp_pop.out_spikes) == 0
data = rand(sz)
maxval = max(data...)
generate_input_spikes!(inp_pop, data, maxval)
@test length(inp_pop.out_spikes) > 0


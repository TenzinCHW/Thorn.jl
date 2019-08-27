id = 1
sz = 5
lr = 0.1
weights = 5 * ones(sz, sz)
spiketype = LIFSpike
inp_pop = PoissonInpPopulation(id, sz, spiketype)
# Input population test
@test length(inp_pop.out_spikes) == 0
proc_pop = LIFPopulation(id, sz, stdp, lr)
data = ones(sz)
maxval = max(data...)
generate_input_spikes!(inp_pop, data, maxval)
@test length(inp_pop.out_spikes) > 0
for s in inp_pop.out_spikes
    process_spike!(proc_pop, weights[:, Int(s.neuron_index)], s)
end
# Processing population test
voltages = proc_pop.u
@test all(voltages .!= 0)
@test length(proc_pop.out_spikes) == 0
num_spike = sum(voltages .> proc_pop.thresh)
output_spike!(proc_pop, inp_pop.out_spikes[end], inp_pop.out_spikes[1])
@test length(proc_pop.out_spikes) == num_spike


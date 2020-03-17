id = 1
sz = 5
numsample = 4
lr = 0.1
weights = 5 * ones(sz, sz)
spiketype = LIFSpike
inp_pop = PoissonInpPopulation(id, sz, spiketype)
proc_pop = LIFPopulation(id, sz, stdp, lr)
data = ones(sz, numsample)
maxval = max(data...)

# Input population test
@test length(inp_pop.out_spikes) == 0
generate_input_spikes!(inp_pop, data, maxval)
@test length(inp_pop.out_spikes) > 0
for s in inp_pop.out_spikes
    @views update_state!(proc_pop, weights[:, Int(s.neuron_id)], s)
end

# Processing population test
voltages = proc_pop.u
@test all(voltages .!= 0)
@test length(proc_pop.out_spikes) == 0
num_spike = sum(voltages .> proc_pop.thresh)
outspikes = Spike[]
output_spike!(outspikes, proc_pop, inp_pop.out_spikes[1])
@test length(outspikes) == num_spike


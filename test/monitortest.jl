function getu(c, ps)
    out = Dict()
    for pop in c.processing_populations
        out[pop.id] = pop.u[:]
    end
    out
end

extractors = Dict("weights"=>(c, ps)->c.weights,
                  "u"=>getu)

input_neuron_types = [(PoissonInpPopulation, 5)]
neuron_types = [(LIFPopulation, 5, stdp, 3.)]
conn = Dict(1=>2)
spiketype = LIFSpike
wt_init() = 5 * rand()
wt_init(m, n) = 30 * rand(m, n)
cortex = Cortex(input_neuron_types, neuron_types, conn, wt_init, spiketype)
data = [rand(sz)]
spikes, monitor = process_sample!(cortex, data, 1., extractors)

import InteractiveUtils
floatarraytypes(dim) = [Array{T, dim} for T in InteractiveUtils.subtypes(AbstractFloat)] 
isfloatarray(arr, dim) = any(isa.([arr], floatarraytypes(dim)))

weights = monitor["weights"][1=>2]
u = monitor["u"][2]
@test isfloatarray(weights, 3)
@test isfloatarray(u, 2)

indices = findall(s->first(s) == 1, spikes)
inp_spikes = last.(spikes[indices])
plot_u = u[:, indices]
f = cortex.processing_populations[1].u_func
x, y = gridify(plot_u[1,:], f, inp_spikes, 1., 1000.)
@test isfloatarray(x, 1)
@test isfloatarray(y, 1)

x, y = rasterspikes(spikes, cortex)
@test all(isfloatarray.(x, 1))
@test all(isfloatarray.(y, 1))


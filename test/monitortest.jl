function getu(c, ps)
    out = Dict()
    for pop in c.processing_populations
        out[pop.id] = [n.u for n in pop.neurons]
    end
    out
end

function getuf(c, ps)
    out = Dict()
    for pop in c.processing_populations
        if first(ps) in population_dependency(c, pop.id)
            out[pop.id] = [(n.u, n.u_func) for n in pop.neurons]
        else
            out[pop.id] = nothing
        end
    end
    out
end

extractors = Dict("weights"=>(c, ps)->c.weights,
                  "u"=>getu,
                  "uf"=>getuf)

input_neuron_types = [(PoissonInpNeuron, 5)]
neuron_types = [(LIFNeuron, 5, stdp, 3.)]
conn = Dict(1=>2)
spiketype = LIFSpike
wt_init() = 5 * rand()
wt_init(m, n) = 5 * rand(m, n)
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

uf = monitor["uf"][2]
inp_spikes = last.(filter(s->first(s) == 1, spikes))
x, y = gridify(uf[1,:], inp_spikes, 1., 1000.)
@test isfloatarray(x, 1)
@test isfloatarray(y, 1)

x, y = rasterspikes(spikes, cortex)
@test all(isfloatarray.(x, 1))
@test all(isfloatarray.(y, 1))


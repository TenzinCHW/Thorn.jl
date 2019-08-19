extractors = Dict("weights"=>c->c.weights,
                  "u"=>c->Dict(pop.id=>[n.u for n in pop.neurons] for pop in c.processing_populations))

input_neuron_types = [(PoissonInpNeuron, 5)]
neuron_types = [(LIFNeuron, 5, stdp, 3.)]
conn = Dict(1=>2)
spiketype = LIFSpike
wt_init() = 5 * rand()
wt_init(m, n) = 5 * rand(m, n)
cortex = Cortex(input_neuron_types, neuron_types, conn, wt_init, spiketype)
data = [rand(sz)]
_, monitor = process_sample!(cortex, data, 1., extractors)

import InteractiveUtils
@test any(isa.([collapse(monitor["weights"])[1=>2]], [Array{T, 3} for T in InteractiveUtils.subtypes(AbstractFloat)]))
@test any(isa.([collapse(monitor["u"])[2]], [Array{T, 2} for T in InteractiveUtils.subtypes(AbstractFloat)]))


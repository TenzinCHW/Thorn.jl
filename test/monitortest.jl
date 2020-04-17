spikesextractorname = "spikes"
extractors = Dict("weights"=>getweights!,
                  "u"=>getu!,
                  spikesextractorname=>getspike!)

wt_init() = 5 * rand()
wt_init(m, n) = 30 * rand(m, n)
cortex = createcortex(;wt_init=wt_init)
numsample = 4
data = getrandomdata(cortex, numsample)
monitor = runprocesssample(cortex, data; extractors=extractors)

@test isa(spikesfromrecord(monitor, spikesextractorname), Vector{Spike})
import InteractiveUtils
floatarraytypes(dim) = [Array{T, dim} for T in InteractiveUtils.subtypes(AbstractFloat)] 
isfloatarray(arr, dim) = any(isa.([arr], floatarraytypes(dim)))

w_spikes, weights = monitor["weights"][1=>3]
u_spikes, u = monitor["u"][3]
@test isfloatarray(weights, 3) # weights is 2D but weights over each spike timing is 3D
@test isfloatarray(u, 2)

#indices = findall(s->first(s) == 1, spikes)
#inp_spikes = last.(spikes[indices])

plot_u = u[1,:] # Last dimension is time
f = cortex.processing_populations[1].u_func
x, y = gridify(plot_u, f, u_spikes, 1., 1000.)
@test length(x) == length(y)
@test isfloatarray(x, 1)
@test isfloatarray(y, 1)

x, y = rasterspikes(cortex)
@test all(isfloatarray.(x, 1))
@test all(isfloatarray.(y, 1))


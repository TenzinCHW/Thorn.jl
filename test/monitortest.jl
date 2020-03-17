function getu(record, cortex, spike)
    dependent_ids = dependent_populations(cortex, spike.pop_id)
    for pop_id in dependent_ids
        push!(record[pop_id], (spike, cortex.populations[pop_id].u[:]))
    end
end

function getweights(record, cortex, spike)
    dependent_ids = dependent_populations(cortex, spike.pop_id)
    for (p2p, w) in cortex.weights
        if last(p2p) in dependent_ids
            push!(record[p2p], (spike, deepcopy(w)))
        end
    end
end

extractors = Dict("weights"=>getweights,
                  "u"=>getu)

sz = 5
numsample = 4
wt_init() = 5 * rand()
wt_init(m, n) = 30 * rand(m, n)
cortex = createcortex(;wt_init=wt_init)
data = [rand(sz, numsample)]
monitor = process_sample!(cortex, data, 1., extractors)

import InteractiveUtils
floatarraytypes(dim) = [Array{T, dim} for T in InteractiveUtils.subtypes(AbstractFloat)] 
isfloatarray(arr, dim) = any(isa.([arr], floatarraytypes(dim)))

w_spikes, weights = monitor["weights"][1=>2]
u_spikes, u = monitor["u"][2]
@test isfloatarray(weights, 3)
@test isfloatarray(u, 2)

#indices = findall(s->first(s) == 1, spikes)
#inp_spikes = last.(spikes[indices])

plot_u = u[1,:]
f = cortex.processing_populations[1].u_func
x, y = gridify(plot_u, f, u_spikes, 1., 1000.)
@test length(x) == length(y)
@test isfloatarray(x, 1)
@test isfloatarray(y, 1)

x, y = rasterspikes(cortex)
@test all(isfloatarray.(x, 1))
@test all(isfloatarray.(y, 1))


ms = .001

compute_rate(pop::InputPopulation, maxval, val) = (val / maxval * (pop.maxrate - pop.minrate) + pop.minrate) * ms

struct ThornInpFuncs
    generate_input_spikes!::Function
    reset!::Function
end


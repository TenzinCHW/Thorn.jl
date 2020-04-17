ms = .001

compute_rate(pop::InputPopulation, val) = (val / maxval * (pop.maxrate - pop.minrate) + pop.minrate) * ms


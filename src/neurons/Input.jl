compute_rate(pop::InputPopulation, maxval, val) = (val / maxval * (pop.maxrate - pop.minrate) + pop.minrate) / pop.sampleperiod


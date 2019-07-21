struct Cortex{T<:AbstractFloat}
    input_populations::Array{InputNeuronPopulation, 1}
    populations::Array{NeuronPopulation, 1}
    pop_connectivity_matrix::Array{Bool, 2}
    interpop_weights::Array{Union{Nothing, Array{T, 2}}, 2}
end

function generate_input_spikes()
    # TODO Given a set of raw sensor data and a Cortex, generate input spikes using the generate_static_input function of each corresponding InputNeuronPopulation
    # Length of array of raw data must match corresponding InputNeuronPopulation
    # If raw sensor data is 2D, the first dimension must represent space and second is time
    # Call generate_static_input for each vector of inputs in second dim, broadcast across the first dimension
    # Must sort the out_spikes property of each InputNeuronPopulation after generating the spikes
end

function process_sample()
    # TODO While any population in a Cortex has spikes in its out_spikes property, call process_next_spike
end

function process_next_spike(cortex::Cortex)
    # TODO Take the head spike from out_spikes queue of each population with smallest time property
    # Route this spike to the correct populations with their weights using the process_spike function for every population that the spike is routed to
    # Call output_spikes to generate output spikes for each of those populations
end

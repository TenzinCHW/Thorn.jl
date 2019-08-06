module PoissonInput
    include("../neuron.jl")
    include("../spike.jl")

    # Rates are in Hz
    maxrate = 20.
    minrate = 1.
    sampleperiod = 1000. # Milliseconds

    struct NeuronStruct{T<:AbstractFloat} <: InputNeuron
        id::UInt
        maxrate::T
        minrate::T
        sampleperiod::T

        function NeuronStruct(id::Int)
            NeuronStruct(UInt(id), maxrate, minrate, sampleperiod)
        end

        function NeuronStruct(id::Int, maxrate, minrate, sampleperiod)
            new{typeof(maxrate)}(UInt(id), maxrate, minrate, sampleperiod)
        end
    end

    function generate_input(neuron::NeuronStruct{T}, maxval::T, sensor_inp::T, spiketype) where T<:AbstractFloat
        time = 0
        spikes = Array{Spike, 1}
        rate = compute_rate(maxval, sensor_inp)
        while (time < neuron.sampleperiod)
            dt = exprand(rate)
            time += dt
            push!(spikes, spiketype(neuron.id, dt))
        end
        spikes
    end

    compute_rate(maxval, val) = val / maxval * (maxrate - minrate) + minrate

    exprand(rate) = - log(1 - rand()) / rate
end

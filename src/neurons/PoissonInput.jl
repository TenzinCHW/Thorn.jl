module PoissonInput
    include("neuron.jl")

    # Rates are in Hz
    maxrate = 20.
    minrate = 1.
    sampleperiod = 1000. # Milliseconds

    struct NeuronStruct{T<:AbstractFloat} <: InputNeuron
        maxrate::T
        minrate::T
        sampleperiod::T

        function NeuronStruct()
            new{typeof(maxrate)}(maxrate, minrate, sampleperiod)
        end

        function NeuronStruct(maxrate, minrate, sampleperiod)
            new{typeof(maxrate)}(maxrate, minrate, sampleperiod)
        end
    end

    function generate_input(neuron::NeuronStruct{T}, sensor_inp::T) where T
    end

    export NeuronStruct
end

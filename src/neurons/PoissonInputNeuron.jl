module PoissonInput
    include("neuron.jl")

    struct NeuronStruct{T<:AbstractFloat} <: InputNeuron
    end

    function generate_input(neuron::NeuronStruct{T}, sensor_inp::T) where T
    end
end

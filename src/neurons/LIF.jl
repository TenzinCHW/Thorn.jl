module LIFNeuron
    include("neuron.jl")
    include("../spike.jl")

    mutable struct NeuronStruct{T<:AbstractFloat} <: ProcessingNeuron
        id::UInt
        u::T
        rest_u::T
        reset_u::T
        tau::T
        thresh::T
        resist::T
        last_out::Spike

        function NeuronStruct(id::Int, u::T, rest_u::T, reset_u::T tau::T, thresh::T, resist::T) where T<:AbstractFloat
            new(UInt(id), u, rest_u, reset_u, tau, thresh, resist, nothing)
        end
    end

    # TODO Find good default values for reset_u, tau, thresh and resist
    def_u = 0
    def_rest_u = 0
    def_reset_u = -5
    def_tau = 1
    def_thresh = 5
    def_resist = 0.5
    NeuronStruct(id::Int) = NeuronStruct(id, def_u, def_rest_u, def_reset_u, def_tau, def_thresh, def_resist, nothing)

    function state_update(neuron::NeuronStruct, weight::AbstractFloat, spike::Spike, prev_spike::Spike)
        decayed = e ^ - (spike.time - prev_spike.time)
        if (neuron.u >= neuron.rest_u)
            decayed = - decayed
        end
        neuron.u = weight + decayed
    end

    function output_spike(neuron::NeuronStruct, spike::Spike, next_spike::Spike)
        if (neuron.u >= neuron.thresh && spike.time <= next_spike.time)
            neuron.u = neuron.reset_u
            neuron.last_out = Spike(neuron.id, spike.time)
            neuron.last_out
        end
    end
end

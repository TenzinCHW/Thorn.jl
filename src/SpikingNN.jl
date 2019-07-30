module SpikingNN
    include("spike.jl")
    include("neurons/neuron.jl")

    export Spike, LIFSpike
    export Neuron
end

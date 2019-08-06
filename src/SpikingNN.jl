module SpikingNN
    include("spike.jl")
    include("neuron.jl")
    include("neurons/LIF.jl")
    include("neurons/PoissonInput.jl")
    include("structure/population.jl")
    include("structure/cortex.jl")
    include("learning/stdp.jl")

    export Spike, LIFSpike
    export Neuron, ProcessingNeuron, InputNeuron
    export LIF
end

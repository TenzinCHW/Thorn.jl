# This is just a custom type for storing any type of weight
abstract type Weights end

# These are for using the dot operator on a Weight
Base.length(w::Weights) = 1

Base.iterate(w::Weights) = w, nothing
Base.iterate(w::Weights, n::Nothing) = nothing


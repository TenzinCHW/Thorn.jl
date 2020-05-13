# This is just a custom type for storing any type of weight
abstract type Weights end

Base.size(w::Weights) = 1 # For broadcasting


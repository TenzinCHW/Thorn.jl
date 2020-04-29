# This is just a custom type for storing any type of weight
struct Weights
    value
end

Base.size(w::Weights) = 1


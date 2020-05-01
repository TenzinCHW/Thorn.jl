# This is just a custom type for storing any type of weight
struct Weights
    value
    weight_update::Function
    lr::AbstractFloat
end

Base.size(w::Weights) = 1


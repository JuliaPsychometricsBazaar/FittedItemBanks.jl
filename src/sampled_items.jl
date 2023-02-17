struct DichotomousSampledItemBank{I} <: AbstractItemBank
    evalpoints::Vector{Float64}
    occs::Array{Float64, 2}
end

struct SampledItemBank <: AbstractItemBank
    evalpoints::Vector{Float64}
    occs::VectorOfArrays{Float64, 2}
end

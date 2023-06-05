using StaticArrays

PerRankReal = AbstractArray{<: AbstractArray{<: Real}, 1}
PerCategoryFloat = AbstractArray{<: AbstractArray{Float64}, 1}

"""
This item bank implements the nominal model. The Graded Partial Credit Model
(GPCM) is implemented in terms of this one. See:

*A Generalized Partial Credit Model: Application of an EM Algorithm*
Muraki, E., (1992).
Applied Psychological Measurement
10.1177/014662169201600206

And/or

*A Generalized Partial Credit Model*
Muraki, E. (1997). 
In Handbook of Modern Item Response Theory.
Springer, New York, NY.
https://doi.org/10.1007/978-1-4757-2691-6_9

Currently, this item bank only supports the normal scaled logistic as the
characteristic/transfer function.
"""
struct NominalItemBank{RankStorageT <: PerRankReal, CategoryStorageT <: PerCategoryFloat} <: AbstractItemBank
    ranks::RankStorageT # ak_1 ... ak_k
    discriminations::Matrix{Float64} # a_1 ... a_n
    cut_points::CategoryStorageT # d_1 ... d_k
end

function NominalItemBank(ranks::Matrix{Float64}, discriminations::Matrix{Float64}, cut_points::Matrix{Float64})
    NominalItemBank(nestedview(ranks), discriminations, nestedview(cut_points))
end

function NominalItemBank(ranks, discriminations::Vector{Float64}, cut_points)
    NominalItemBank(ranks, reshape(discriminations, 1, :), cut_points)
end

function GPCMItemBank(discriminations, cut_points::PerCategoryFloat)
    NominalItemBank(
        # XXX: Could probably be more efficient by making this lazy somehow
        [1:length(item_cut_points) for item_cut_points in cut_points],
        discriminations,
        cut_points
    )
end

function GPCMItemBank(discriminations, cut_points::Matrix{Float64})
    GPCMItemBank(discriminations, nextedview(cut_points))
end

DomainType(::NominalItemBank) = VectorContinuousDomain()
ResponseType(::NominalItemBank) = MultinomialResponse()

Base.length(item_bank::NominalItemBank) = size(item_bank.discriminations, 2)
domdims(item_bank::NominalItemBank) = size(item_bank.discriminations, 1)

function linears(ir::ItemResponse{<:NominalItemBank}, θ)
    aks = ir.item_bank.ranks[ir.index]
    as = @view ir.item_bank.discriminations[:, ir.index]
    ds = ir.item_bank.cut_points[ir.index]
    aks .* (dot(as, θ) .+ ds)
end

function (ir::ItemResponse{<:NominalItemBank})(θ)
    resp(ir, θ)
end

function num_response_categories(ir::ItemResponse{<:NominalItemBank})
    length(ir.item_bank.cut_points[ir.index])
end

function resp_vec(ir::ItemResponse{<:NominalItemBank}, θ)
    ir(θ)
end

function num_ranks(ir::ItemResponse{<:NominalItemBank})
    length(ir.item_bank.ranks[ir.index])
end

function resp(ir::ItemResponse{<:NominalItemBank}, θ)
    outs = StaticArrays.sacollect(SVector{num_ranks(ir), Float64}, exp.(linears(ir, θ)))
    outs ./ sum(outs)
end

function logresp(ir::ItemResponse{<:NominalItemBank}, θ)
    outs = StaticArrays.sacollect(SVector{num_ranks(ir), Float64}, linears(ir, θ))
    outs .= outs - logsumexp(linears(ir, θ))
end

function item_params(item_bank::NominalItemBank, idx)
    (; discrimination=item_bank.discriminations[idx, :])
end
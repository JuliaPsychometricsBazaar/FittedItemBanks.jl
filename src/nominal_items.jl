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
PerCategoryFloat = AbstractArray{<: AbstractArray{Float64}}

struct NominalItemBank{CategoryStorageT <: PerCategoryFloat} <: AbstractItemBank
    ranks::CategoryStorageT # ak_1 ... ak_k
    discriminations::Matrix{Float64} # a_1 ... a_n
    cut_points::CategoryStorageT # d_1 ... d_k
    labels::MaybeLabels
end

function NominalItemBank(ranks::Matrix{Float64}, discriminations::Matrix{Float64}, cut_points::Matrix{Float64}, labels=nothing)
    NominalItemBank(nestedview(ranks), discriminations, nestedview(cut_points), labels)
end

function NominalItemBank(ranks, discriminations::Vector{Float64}, cut_points, labels=nothing)
    NominalItemBank(ranks, reshape(discriminations, 1, :), cut_points, labels)
end

function GPCMItemBank(discriminations, cut_points::PerCategoryFloat, labels=nothing)
    NominalItemBank(
        # XXX: Could probably be more efficient by making this lazy somehow
        [1:length(item_cut_points) for item_cut_points in cut_points],
        discriminations,
        cut_points,
        labels
    )
end

function GPCMItemBank(discriminations, cut_points::Matrix{Float64}, labels=nothing)
    GPCMItemBank(discriminations, nextedview(cut_points), labels)
end

MathTraits.DomainType(::NominalItemBank) = OneDimContinuousDomain()
Responses.ResponseType(::NominalItemBank) = MultinomialResponse()

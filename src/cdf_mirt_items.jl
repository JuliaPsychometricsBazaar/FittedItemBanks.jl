# TODO: Could probably refactor to be more generic w.r.t. cdf_items.jl

using LinearAlgebra: dot

"""
This item bank corresponds to the most commonly found version of MIRT in the
literature. Its items feature multidimensional discriminations and its learners
multidimensional abilities, but item difficulties are single-dimensional.
"""
struct CdfMirtItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    difficulties::Vector{Float64}
    discriminations::Matrix{Float64}

    function CdfMirtItemBank(
        distribution::DistT,
        difficulties::Vector{Float64},
        discriminations::Matrix{Float64},
    ) where {DistT <: ContinuousUnivariateDistribution}
        if size(discriminations, 2) != length(difficulties)
            error(
                "Number of items in first (only) dimension of difficulties " *
                "should match number of item in 2nd dimension of discriminations"
            )
        end
        new{typeof(distribution)}(distribution, difficulties, discriminations)
    end
end

DomainType(::CdfMirtItemBank) = VectorContinuousDomain()
ResponseType(::CdfMirtItemBank) = BooleanResponse()

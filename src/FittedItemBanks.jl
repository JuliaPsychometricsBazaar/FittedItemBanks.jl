"""
This module provides abstract and concrete item banks, which store information
about items and their parameters such as difficulty, most typically resulting
from fitting an Item-Response Theory (IRT) model.
"""
module FittedItemBanks

export AbstractItemBank, GuessItemBank
export SlipItemBank, TransferItemBank,
export ItemBank2PL, ItemBank3PL, ItemBank4PL
export ItemBankMirt2PL, ItemBankMirt3PL, ItemBankMirt4PL

using Distributions
using Random: AbstractRNG
using Distributions: Logistic, UnivariateDistribution, Normal, MvNormal, Zeros, ScalMat
using Lazy: @forward

abstract type AbstractItemBank end

function item_idxs(item_bank::AbstractItemBank)
    Base.OneTo(length(item_bank))
end

# This seems to be the most commonly found exact value in the wild, see e.g. the
# R package `mirt``
const scaling_factor = 1.702

struct NormalScaledLogistic
    inner::Logistic
    NormalScaledLogistic(μ, σ) = Logistic(μ / scaling_factor, σ / scaling_factor)
end

NormalScaledLogistic() = NormalScaledLogistic(0.0, 1.0)

@forward NormalScaledLogistic.inner (
    sampler, pdf, logpdf, cdf, quantile, minimum, maximum, insupport, mean, var,
    modes, mode, skewness, kurtosis, entropy, mgf, cf
)

rand(rng::AbstractRNG, d::NormalScaledLogistic) = rand(rng, d.inner)

abstract type DomainType end
abstract type DiscreteDomain <: DomainType end
abstract type ContinuousDomain <: DomainType end
struct VectorContinuousDomain <: ContinuousDomain end
struct OneDimContinuousDomain <: ContinuousDomain end
struct DiscreteIndexableDomain <: DiscreteDomain end
struct DiscreteIterableDomain <: DiscreteDomain end

abstract type ResponseType end
struct BooleanResponse <: ResponseType end
struct MultinomialResponse <: ResponseType end

include("./guess_slip_items.jl")
include("./cdf_items.jl")
include("./cdf_mirt_items.jl")
include("./sampled_items.jl")
include("./nominal_items.jl")
include("./porcelain.jl")

end

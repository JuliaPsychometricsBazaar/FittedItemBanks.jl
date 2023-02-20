"""
This module provides abstract and concrete item banks, which store information
about items and their parameters such as difficulty, most typically resulting
from fitting an Item-Response Theory (IRT) model.
"""
module FittedItemBanks

export AbstractItemBank, GuessItemBank
export SlipItemBank, TransferItemBank
export ItemBank2PL, ItemBank3PL, ItemBank4PL
export ItemBankMirt2PL, ItemBankMirt3PL, ItemBankMirt4PL

using Distributions
using Random: AbstractRNG
using Distributions: Logistic, UnivariateDistribution, Normal, MvNormal, Zeros, ScalMat
using Lazy: @forward
using ArraysOfArrays: VectorOfArrays
using PsychometricsBazzarBase.Integrators

abstract type AbstractItemBank end

function Base.eachindex(item_bank::AbstractItemBank)
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
struct ItemResponse{ItemBankT <: AbstractItemBank}
    item_bank::ItemBankT
    index::Int
end

"""
Binary search for the point x where the integral from -inf...x is target += precis
"""
function _search(
    integrator::Integrator,
    ir::F,
    lim_lower,
    lim_upper,
    target,
    precis;
    max_iters=50,
    denom=normdenom(integrator)
) where {F}
    lower = lim_lower
    upper = lim_upper
    @info "max_iters" max_iters
    for _ in 1:max_iters
        pivot = lower + (upper - lower) / 2
        @info "limits" lo=lim_lower hi=pivot
        mass = intval(integrator(ir; lo=lim_lower, hi=pivot))
        ratio = mass / denom
        @info "mass" mass denom ratio target precis
        if target - precis <= ratio <= target
            return pivot
        elseif ratio < target
            lower = pivot
        else
            upper = pivot
        end
    end
    error("Could not find point after $max_iters iterations")
end

"""
Given an item bank, this function returns the domain of the item bank, i.e. the
range (lo, hi) which includes for each item the range in which the the item
response function is changing.
"""
function item_bank_domain(
    integrator::Integrator,
    item_bank::AbstractItemBank;
    tol=1e-3,
    precis=1e-2,
    zero_symmetric=false
)
    tol1 = tol / 2.0
    eff_precis = tol1 * precis
    if length(item_bank) == 0
        (integrator.lo, integrator.hi)
    end
    lo = integrator.hi
    hi = integrator.lo
    for item_idx in item_idxs(item_bank)
        ir = ItemResponse(item_bank, item_idx)
        # XXX: denom should be the denom of the item response
        denom = normdenom(integrator)
        inv_ir(x) = 1.0 - ir(-x)
        if intval(integrator(ir; lo=integrator.lo, hi=lo)) > tol1
            lo = _search(integrator, ir, integrator.lo, lo, tol1, eff_precis; denom=denom)
        end
        inv_denom = integrator.hi - integrator.lo - denom
        # XXX
        if intval(integrator(inv_ir; lo=-integrator.hi, hi=hi)) > tol1
            hi = -_search(integrator, inv_ir, -integrator.hi, -hi, tol1, eff_precis; denom=inv_denom)
        end
    end
    if zero_symmetric
        dist = max(abs(lo), abs(hi))
        (-dist, dist)
    else
        (lo, hi)
    end
end


include("./guess_slip_items.jl")
include("./cdf_items.jl")
include("./cdf_mirt_items.jl")
include("./sampled_items.jl")
include("./nominal_items.jl")
include("./porcelain.jl")

end

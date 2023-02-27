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

export Smoother, KernelSmoother, DichotomousSmoothedItemBank, DichotomousPointsItemBank
export ItemResponse, resp, resp_vec

export DomainType, DiscreteDomain, ContinuousDomain, VectorContinuousDomain
export OneDimContinuousDomain, DiscreteIndexableDomain, DiscreteIterableDomain

using Distributions
using Random: AbstractRNG
using Distributions: Logistic, UnivariateDistribution, Normal, MvNormal, Zeros, ScalMat
using Lazy: @forward
using ArraysOfArrays: VectorOfArrays
using StaticArrays: SVector
using PsychometricsBazzarBase.ExtraDistributions: NormalScaledLogistic

abstract type AbstractItemBank end

function Base.eachindex(item_bank::AbstractItemBank)
    Base.OneTo(length(item_bank))
end

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
Binary search for the point x where f(x) = target += precis given f is assumed as monotonically increasing.
"""
function _search(
    f::F,
    lim_lower,
    lim_upper,
    target,
    precis;
    max_iters=50,
) where {F}
    lower = lim_lower
    upper = lim_upper
    for _ in 1:max_iters
        pivot = lower + (upper - lower) / 2
        y = f(pivot)
        if target - precis <= y <= target + precis
            return pivot
        elseif y < target
            lower = pivot
        else
            upper = pivot
        end
    end
    error("Could not find point after $max_iters iterations")
end

function search_per_dim(::VectorContinuousDomain, ir, lo, hi, start, target, thresh)
    buf = copy(start)
    bests = Array{Float64}(undef, size(lo))
    for idx in eachindex(lo)
        function partial_deriv(θ)
            buf[idx] = θ
            max(abs.(resp_vec(ir, buf) .- target))
        end
        bests[idx] = _search(partial_deriv, lo[idx], hi[idx], 0, thresh)
        buf[idx] = start[idx]
    end
    bests
end

function search_per_dim(::OneDimContinuousDomain, ir, lo, hi, start, target, thresh)
    function partial_deriv(θ)
        maximum(abs.(resp_vec(ir, θ) .- target))
    end
    _search(partial_deriv, lo, hi, 0, thresh)
end

"""
Given an item bank, this function returns the domain of the item bank, i.e. the
range (lo, hi) which includes for each item the range in which the the item
response function is changing.
"""
function item_bank_domain(
    item_bank::AbstractItemBank;
    zero_symmetric=false,
    items=eachindex(item_bank),
    thresh=nothing
)
    if length(item_bank) == 0
        (NaN, NaN)
    end
    cur_lo = Inf
    cur_hi = -Inf
    for item_idx in items
        ir = ItemResponse(item_bank, item_idx)
        if thresh === nothing
            item_lo, item_hi = item_domain(ir)
        else
            item_lo, item_hi = item_domain(ir, thresh)
        end
        if item_lo < cur_lo
            cur_lo = item_lo
        end
        if item_hi > cur_hi
            cur_hi = item_hi
        end
        #cur_lo = search_per_dim(DomainType(item_bank), ir, lo, cur_lo, cur_lo, minabilresp(ir), thresh)
        #cur_hi = search_per_dim(DomainType(item_bank), ir, cur_hi, hi, cur_hi, maxabilresp(ir), thresh)
    end
    if zero_symmetric
        dist = max(abs(cur_lo), abs(cur_hi))
        (-dist, dist)
    else
        (cur_lo, cur_hi)
    end
end


include("./guess_slip_items.jl")
include("./cdf_items.jl")
include("./cdf_mirt_items.jl")
include("./sampled_items.jl")
include("./nominal_items.jl")
include("./porcelain.jl")

end

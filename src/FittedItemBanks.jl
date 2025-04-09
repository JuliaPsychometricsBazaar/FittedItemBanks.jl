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
export NominalItemBank, GPCMItemBank
export MonopolyItemBank
export BSplineItemBank

export Smoother, KernelSmoother, NearestNeighborSmoother
export PointsItemBank
export DichotomousSmoothedItemBank, DichotomousPointsItemBank,
       MultiGridDichotomousPointsItemBank
export OneDimensionItemBankAdapter

export domdims, item_bank_domain
export ItemResponse, resp, resp_vec, responses, item_params
export num_response_categories

export spec_description_slug, spec_description_short, spec_description_long

export DomainType, DiscreteDomain, ContinuousDomain, VectorContinuousDomain
export OneDimContinuousDomain, DiscreteIndexableDomain, DiscreteIterableDomain

export ResponseType, BooleanResponse, MultinomialResponse

export SimpleItemBankSpec, StdModel2PL, StdModel3PL, StdModel4PL

using Distributions
using Distributions: Logistic, UnivariateDistribution, Normal, MvNormal, Zeros, ScalMat
using Lazy: @forward
using ArraysOfArrays: VectorOfArrays, nestedview
using StaticArrays: SVector
using PsychometricsBazaarBase.ConstDistributions: normal_scaled_logistic
using PsychometricsBazaarBase.Interpolators: interp
using DocStringExtensions
using ArraysOfArrays
using BSplines
using BSplines: NoDerivative, bsplines_destarray, _bsplines!, bsplines_offsetarray
using LogExpFunctions
using ResumableFunctions
using Polynomials

const default_mass = 1e-2

"""
$(TYPEDEF)

Base supertype for all item banks.
"""
abstract type AbstractItemBank end

# This is used for dummy methods to document interfaces.
struct _DocsItemBank
    _DocsItemBank(::_DocsItemBank) = nothing
end

"""
$(TYPEDSIGNATURES)

Returns an `AbstractUnitRange` of item indices for the item bank.
"""
function Base.eachindex(item_bank::AbstractItemBank)
    Base.OneTo(length(item_bank))
end

"""
```julia
$(FUNCTIONNAME)(item_bank::AbstractItemBank)
```

Returns the number of items in the item bank.
"""
function Base.length(::_DocsItemBank) end

"""
$(TYPEDSIGNATURES)

Returns the raw parameters for the item at `idx` as a named tuple.
This may return nothing for some item banks.
This is debugging/informational use only.
Use (ItemResponse)[@ref] for actual item response functions.
"""
function item_params(item_bank::AbstractItemBank, idx)
    (;)
end

"""
$(TYPEDEF)

```julia
$(FUNCTIONNAME)(::AbstractItemBank) -> DomainType
```

Domain type for a item banks' item response functions. Used as a trait.
"""
abstract type DomainType end

"""
$(TYPEDEF)

A discrete domain. Typically this is a sampled version of a continuous domain
item bank.

Item response functions with discrete domains tend to support less operations
than those with continuous domains.
"""
abstract type DiscreteDomain <: DomainType end

"""
$(TYPEDEF)

A continuous domain.
"""
abstract type ContinuousDomain <: DomainType end

"""
$(TYPEDEF)

A continuous domain that is vector valued.
"""
struct VectorContinuousDomain <: ContinuousDomain end

"""
$(TYPEDEF)

A continuous domain that is scalar valued.
"""
struct OneDimContinuousDomain <: ContinuousDomain end

"""
$(TYPEDEF)

An discrete domain which is efficiently indexable and iterable.
"""
struct DiscreteIndexableDomain <: DiscreteDomain end

"""
$(TYPEDEF)

An discrete domain which is only efficiently iterable.
"""
struct DiscreteIterableDomain <: DiscreteDomain end

"""
$(TYPEDEF)

```julia
$(FUNCTIONNAME)(::AbstractItemBank) -> ResponseType
```

A response type for an item bank. Used as a trait.
"""
abstract type ResponseType end

"""
$(TYPEDEF)

A boolean/dichotomous response.
"""
struct BooleanResponse <: ResponseType end

"""
$(TYPEDEF)

A multinomial response, including ordinal responses.
"""
struct MultinomialResponse <: ResponseType end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
$(TYPEDSIGNATURES)

An item response.
"""
struct ItemResponse{ItemBankT <: AbstractItemBank}
    item_bank::ItemBankT
    index::Int
end

# Mark as a scalar for broadcasting
Base.broadcastable(ir::ItemResponse) = Ref(ir)

"""
$(TYPEDSIGNATURES)

Returns an `AbstractVector` of possible outcomes for a given (ItemResponse)[@ref].
"""
function responses(ir::ItemResponse)
    responses(ResponseType(ir.item_bank), ir)
end

function responses(::BooleanResponse, ir::ItemResponse)
    SVector(false, true)
end

function responses(::MultinomialResponse, ir::ItemResponse)
    1:num_response_categories(ir)
end

function spec_description_short(val)
    spec_description(val, :short)
end

function spec_description_long(val)
    spec_description(val, :long)
end

function spec_description_slug(val)
    spec_description(val, :slug)
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
        max_iters = 50
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
$(TYPEDSIGNATURES)

Given an item bank, this function returns the domain of the item bank, i.e. the
range (lo, hi) which includes for each item the range in which the the item
response function is changing.
"""
function item_bank_domain(
        item_bank::AbstractItemBank;
        zero_symmetric = false,
        items = eachindex(item_bank),
        thresh = nothing
)
    return item_bank_domain(
        DomainType(item_bank),
        item_bank;
        zero_symmetric = zero_symmetric,
        items = items,
        thresh = thresh
    )
end

function item_bank_domain(
        ::OneDimContinuousDomain,
        item_bank::AbstractItemBank;
        zero_symmetric = false,
        items = eachindex(item_bank),
        thresh = nothing
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
            item_lo, item_hi = item_domain(ir; mass = thresh)
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

function item_bank_domain(
        ::VectorContinuousDomain,
        item_bank::AbstractItemBank;
        zero_symmetric = false,
        items = eachindex(item_bank),
        thresh = nothing,
        reference_point = nothing
)
    ndims = domdims(item_bank)
    if reference_point === nothing
        reference_point = zeros(ndims)
    end
    cur_lo = fill(Inf, ndims)
    cur_hi = fill(-Inf, ndims)
    for item_idx in items
        ir = ItemResponse(item_bank, item_idx)
        if thresh === nothing
            item_lo, item_hi = item_domain(ir; reference_point = reference_point)
        else
            item_lo, item_hi = item_domain(
                ir; reference_point = reference_point, mass = thresh)
        end
        for idx in 1:ndims
            if item_lo[idx] < cur_lo[idx]
                cur_lo[idx] = item_lo[idx]
            end
            if item_hi[idx] > cur_hi[idx]
                cur_hi[idx] = item_hi[idx]
            end
        end
    end
    if zero_symmetric
        dist = max.(abs.(cur_lo), abs.(cur_hi))
        (-dist, dist)
    else
        (cur_lo, cur_hi)
    end
end

VectorOfVectorsFloat64 = VectorOfVectors{
    Float64, Vector{Float64}, Vector{Int64}, Vector{Tuple{}}}

"""
```julia
function $(FUNCTIONNAME)(item_bank::AbstractItemBank, idxs)
```

Return a new item bank of the same type, with the items at the given indices.
"""
function subset end

"""
```julia
$(FUNCTIONNAME)(ir::ItemResponse, θ) -> Float64  # For BooleanResponse item banks only
$(FUNCTIONNAME)(ir::ItemResponse, outcome, θ) -> Float64
```

Return the value of the item response outcome function for the item response
`ir`, the outcome `outcome` and the ability values `θ`.
For `BooleanResponse` item banks, `outcome` can be omitted in which case the
outcome is assumed to be `true`.
"""
function resp end

"""
```julia
$(FUNCTIONNAME)(ir::ItemResponse, θ) -> AbstractVector{Float64}
```

Return the vector value of the item response function for the item response
`ir`, the outcome `outcome` and the ability values `θ`.

The outcome at each index corresponds with the indices returned by the [responses](@ref) function.
"""
function resp_vec end

include("./adapter.jl")
include("./guess_slip_items.jl")
include("./cdf_items.jl")
include("./cdf_mirt_items.jl")
include("./monopoly.jl")
include("./bspline.jl")
include("./sampled_items.jl")
include("./nominal_items.jl")
include("./porcelain.jl")
include("./DummyData/DummyData.jl")
include("./precompiles.jl")

end

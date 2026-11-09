# Keep both ordinary tails: deriving a small CCDF as 1-CDF loses it when
# the CDF rounds to one. Other distributions retain their own tail algorithms.
@inline _cdf_response_pair(distribution, x) =
    SVector(ccdf(distribution, x), cdf(distribution, x))

@inline _logistic_coordinate(x, location, scale) = (x - location) / scale

# Dual-number scales must retain their derivative even when their value is one.
@inline function _logistic_coordinate(x::T, location::T, scale::T) where {
        T <: Union{Float32, Float64}}
    shifted = x - location
    isone(scale) ? shifted : shifted / scale
end

@inline function _cdf_response_pair(distribution::Logistic, x)
    location, scale = params(distribution)
    z = _logistic_coordinate(x, location, scale)
    positive = z >= zero(z)
    # Expose Base's exp implementation to vectorization of the caller's loop.
    e = @inline exp(ifelse(positive, -z, z))
    large = inv(one(e) + e)
    small = e * large
    SVector(ifelse(positive, small, large), ifelse(positive, large, small))
end

"""
```julia
struct TransferItemBank <: AbstractItemBank
$(FUNCTIONNAME)(distribution, difficulties, discriminations) -> $(FUNCTIONNAME)
DomainType(::TransferItemBank) = OneDimContinuousDomain()
ResponseType(::TransferItemBank) = BooleanResponse()
```

This item bank corresponds to a 2 parameter, single dimensional IRT model.
"""
struct TransferItemBank{DistT <: ContinuousUnivariateDistribution, ParamVecT <: AbstractVector{<:Number}} <: AbstractItemBank
    distribution::DistT
    difficulties::ParamVecT
    discriminations::ParamVecT
end

DomainType(::TransferItemBank) = OneDimContinuousDomain()
ResponseType(::TransferItemBank) = BooleanResponse()
Base.eltype(::TransferItemBank{_, ParamT}) where {_, ParamT} = ParamT

function Base.length(item_bank::TransferItemBank)
    length(item_bank.difficulties)
end

function subset(item_bank::TransferItemBank, idxs)
    TransferItemBank(
        item_bank.distribution,
        item_bank.difficulties[idxs],
        item_bank.discriminations[idxs]
    )
end
@views function subset_view(item_bank::TransferItemBank, idxs)
    TransferItemBank(
        item_bank.distribution,
        item_bank.difficulties[idxs],
        item_bank.discriminations[idxs]
    )
end

domdims(item_bank::TransferItemBank) = 0

function _norm_abil_1d(θ, difficulty, discrimination)
    (θ - difficulty) * discrimination
end

function _unnorm_abil_1d(θ, difficulty, discrimination)
    θ / discrimination + difficulty
end

function norm_abil(ir::ItemResponse{<:TransferItemBank}, θ)
    _norm_abil_1d(
        θ, ir.item_bank.difficulties[ir.index], ir.item_bank.discriminations[ir.index])
end

function unnorm_abil(ir::ItemResponse{<:TransferItemBank}, θ)
    _unnorm_abil_1d(
        θ, ir.item_bank.difficulties[ir.index], ir.item_bank.discriminations[ir.index])
end

@inline function resp_vec(ir::ItemResponse{<:TransferItemBank}, θ)
    _cdf_response_pair(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_response_category_uncertain(ir::ItemResponse{<:TransferItemBank}, outcome;
        mass = default_mass)
    if outcome
        lo = unnorm_abil(ir, quantile(ir.item_bank.distribution, 1.0 - mass))
        return lo..Inf
    else
        hi = unnorm_abil(ir, quantile(ir.item_bank.distribution, mass))
        return -Inf..hi
    end
end

function maxabilresp(::ItemResponse{<:TransferItemBank})
    return SVector(0.0, 1.0)
end

function minabilresp(::ItemResponse{<:TransferItemBank})
    return SVector(1.0, 0.0)
end

@inline function resp(ir::ItemResponse{<:TransferItemBank}, outcome::Bool, θ)
    if outcome
        resp(ir, θ)
    else
        cresp(ir, θ)
    end
end

function resp(ir::ItemResponse{<:TransferItemBank}, θ)
    cdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function density(ir::ItemResponse{<:TransferItemBank}, θ)
    pdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function cresp(ir::ItemResponse{<:TransferItemBank}, θ)
    ccdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_params(item_bank::TransferItemBank, idx)
    (; difficulty = item_bank.difficulties[idx],
        discrimination = item_bank.discriminations[idx])
end

function convert_parameter_type(T::Type, item_bank::TransferItemBank{DistT}) where {DistT}
    TransferItemBank(
        convert(constructorof(DistT){T}, item_bank.distribution),
        convert(Vector{T}, item_bank.difficulties),
        convert(Vector{T}, item_bank.discriminations)
    )
end

num_response_categories(ir::ItemResponse{<:TransferItemBank}) = 2

function log_resp_vec(ir::ItemResponse{<:TransferItemBank}, θ)
    nθ = norm_abil(ir, θ)
    return SVector(
        logccdf(ir.item_bank.distribution, nθ),
        logcdf(ir.item_bank.distribution, nθ)
    )
end

function log_resp(ir::ItemResponse{<:TransferItemBank}, outcome::Bool, θ)
    nθ = norm_abil(ir, θ)
    if outcome
        logcdf(ir.item_bank.distribution, nθ)
    else
        logccdf(ir.item_bank.distribution, nθ)
    end
end

log_resp(ir::ItemResponse{<:TransferItemBank}, θ) = log_resp(ir, true, θ)

"""
```julia
struct $(FUNCTIONNAME) <: AbstractItemBank
$(FUNCTIONNAME)(distribution, difficulties, discriminations) -> $(FUNCTIONNAME)
DomainType(::$(FUNCTIONNAME)) = OneDimContinuousDomain()
ResponseType(::$(FUNCTIONNAME)) = BooleanResponse()
```

This item bank corresponds the slope-intercept form of teh 2 parameter, single
dimensional IRT model.
"""
struct SlopeInterceptTransferItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    intercepts::Vector{Float64}
    slopes::Vector{Float64}
end

function SlopeInterceptTransferItemBank(item_bank::TransferItemBank)
    SlopeInterceptTransferItemBank(
        item_bank.distribution,
        item_bank.difficulties .* item_bank.discriminations,
        item_bank.discriminations
    )
end

function TransferItemBank(item_bank::SlopeInterceptTransferItemBank)
    TransferItemBank(
        item_bank.distribution,
        item_bank.intercepts ./ item_bank.slopes,
        item_bank.slopes
    )
end

DomainType(::SlopeInterceptTransferItemBank) = OneDimContinuousDomain()
ResponseType(::SlopeInterceptTransferItemBank) = BooleanResponse()
domdims(item_bank::SlopeInterceptTransferItemBank) = 0

function Base.length(item_bank::SlopeInterceptTransferItemBank)
    length(item_bank.intercepts)
end

function subset(item_bank::SlopeInterceptTransferItemBank, idxs)
    SlopeInterceptTransferItemBank(
        item_bank.distribution,
        item_bank.intercepts[idxs],
        item_bank.slopes[idxs]
    )
end

@views function subset_view(item_bank::SlopeInterceptTransferItemBank, idxs)
    SlopeInterceptTransferItemBank(
        item_bank.distribution,
        item_bank.intercepts[idxs],
        item_bank.slopes[idxs]
    )
end

function _norm_abil_1d_si(θ, intercept, slope)
    θ * slope - intercept
end

function _unnorm_abil_1d_si(θ, intercept, slope)
    (θ + intercept) / slope
end

function norm_abil(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    _norm_abil_1d_si(
        θ, ir.item_bank.intercepts[ir.index], ir.item_bank.slopes[ir.index])
end

function unnorm_abil(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    _unnorm_abil_1d_si(
        θ, ir.item_bank.intercepts[ir.index], ir.item_bank.slopes[ir.index])
end

@inline function resp_vec(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    _cdf_response_pair(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_response_category_uncertain(ir::ItemResponse{<:SlopeInterceptTransferItemBank},
        outcome; mass = default_mass)
    if outcome
        lo = unnorm_abil(ir, quantile(ir.item_bank.distribution, 1.0 - mass))
        return lo..Inf
    else
        hi = unnorm_abil(ir, quantile(ir.item_bank.distribution, mass))
        return -Inf..hi
    end
end

function maxabilresp(::ItemResponse{<:SlopeInterceptTransferItemBank})
    return SVector(0.0, 1.0)
end

function minabilresp(::ItemResponse{<:SlopeInterceptTransferItemBank})
    return SVector(1.0, 0.0)
end

@inline function resp(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, outcome::Bool, θ)
    if outcome
        resp(ir, θ)
    else
        cresp(ir, θ)
    end
end

function resp(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    cdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function density(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    pdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function cresp(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    ccdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_params(item_bank::SlopeInterceptTransferItemBank, idx)
    (; intercept = item_bank.intercepts[idx],
       slope = item_bank.slopes[idx])
end

num_response_categories(ir::ItemResponse{<:SlopeInterceptTransferItemBank}) = 2

function log_resp_vec(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    nθ = norm_abil(ir, θ)
    return SVector(
        logccdf(ir.item_bank.distribution, nθ),
        logcdf(ir.item_bank.distribution, nθ)
    )
end

function log_resp(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, outcome::Bool, θ)
    nθ = norm_abil(ir, θ)
    if outcome
        logcdf(ir.item_bank.distribution, nθ)
    else
        logccdf(ir.item_bank.distribution, nθ)
    end
end

log_resp(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ) = log_resp(ir, true, θ)

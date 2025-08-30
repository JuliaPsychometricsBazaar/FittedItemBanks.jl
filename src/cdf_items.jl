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

function resp_vec(ir::ItemResponse{<:TransferItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

#=
function density_vec(ir::ItemResponse{<:TransferItemBank}, θ)
    density1 = density(ir, θ)
    SVector(-density1, density1)
end
=#

#=function item_domain(ir::ItemResponse, mass = 1e-3)
    item_domain(ir, mass, mass)
end=#

function item_domain(ir::ItemResponse{<:TransferItemBank};
        mass = default_mass, left_mass = mass, right_mass = mass)
    (
        unnorm_abil(ir, quantile(ir.item_bank.distribution, left_mass)),
        unnorm_abil(ir, quantile(ir.item_bank.distribution, 1.0 - right_mass))
    )
end

function maxabilresp(::ItemResponse{<:TransferItemBank})
    return SVector(0.0, 1.0)
end

function minabilresp(::ItemResponse{<:TransferItemBank})
    return SVector(1.0, 0.0)
end

function resp(ir::ItemResponse{<:TransferItemBank}, outcome::Bool, θ)
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

function resp_vec(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

function item_domain(ir::ItemResponse{<:SlopeInterceptTransferItemBank};
        mass = default_mass, left_mass = mass, right_mass = mass)
    (
        unnorm_abil(ir, quantile(ir.item_bank.distribution, left_mass)),
        unnorm_abil(ir, quantile(ir.item_bank.distribution, 1.0 - right_mass))
    )
end

function maxabilresp(::ItemResponse{<:SlopeInterceptTransferItemBank})
    return SVector(0.0, 1.0)
end

function minabilresp(::ItemResponse{<:SlopeInterceptTransferItemBank})
    return SVector(1.0, 0.0)
end

function resp(ir::ItemResponse{<:SlopeInterceptTransferItemBank}, outcome::Bool, θ)
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

struct TransferItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    difficulties::Vector{Float64}
    discriminations::Vector{Float64}
end

DomainType(::TransferItemBank) = OneDimContinuousDomain()
ResponseType(::TransferItemBank) = BooleanResponse()

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
        left_mass = default_mass, right_mass = default_mass)
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

function spec_description(item_bank::TransferItemBank, level)
    if item_bank.distribution == normal_scaled_logistic
        if level == :long
            return "Two parameter unidimensional item bank with normal scaled logistic distribution"
        elseif level == :short
            return "2PL"
        else
            return "2pl"
        end
    else
        if level == :long
            return "Two parameter unidimensional item bank with unknown transfer function"
        elseif level == :short
            return "2P"
        else
            return "2p"
        end
    end
end

num_response_categories(ir::ItemResponse{<:TransferItemBank}) = 2

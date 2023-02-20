struct TransferItemBank{DistT <: ContinuousUnivariateDistribution} <: AbstractItemBank
    distribution::DistT
    difficulties::Vector{Float64}
    discriminations::Vector{Float64}
end

DomainType(::TransferItemBank) = OneDimContinuousDomain()
ResponseType(::TransferItemBank) = BooleanResponse()

function _norm_abil_1d(θ, difficulty, discrimination)
    (θ - difficulty) * discrimination
end

function norm_abil(ir::ItemResponse{<:TransferItemBank}, θ)
    _norm_abil_1d(θ, ir.item_bank.difficulties[ir.index], ir.item_bank.discriminations[ir.index])
end

function resp_vec(ir::ItemResponse{<:TransferItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
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

function cresp(ir::ItemResponse{<:TransferItemBank}, θ)
    ccdf(ir.item_bank.distribution, norm_abil(ir, θ))
end

function item_params(item_bank::TransferItemBank, idx)
    (; difficulty=item_bank.difficulties[idx], discrimination=item_bank.discriminations[idx])
end
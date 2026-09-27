"""
$(TYPEDEF)

A [`DichotomousPointsItemBank`](@ref) (`inner_bank`) with its tabulated response
probabilities precomputed in log-space (`log_ys`), so that likelihoods over
many responses can be accumulated by summation instead of multiplication.

The cache has dimensions `(response, grid point, item)`, with responses ordered
as `[false, true]`. Access it with [`item_log_ys`](@ref). The cache is computed
at construction; do not mutate the underlying probabilities afterwards.
`resp_vec` evaluates the nearest grid point.
"""
struct DichotomousPointsWithLogsItemBank{DomainT} <: PointsItemBank
    inner_bank::DichotomousPointsItemBank{DomainT}
    log_ys::Array{Float64, 3}
end

function DichotomousPointsWithLogsItemBank(inner_bank::DichotomousPointsItemBank)
    ys = inner_bank.ys
    log_ys = stack((log1p.(-ys), log.(ys)); dims = 1)
    return DichotomousPointsWithLogsItemBank(inner_bank, log_ys)
end

function inner_item_response(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank})
    ItemResponse(ir.item_bank.inner_bank, ir.index)
end

@forward DichotomousPointsWithLogsItemBank.inner_bank Base.length,
domdims, item_bank_xs, ResponseType, DomainType

function item_domain(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank}; kwargs...)
    item_domain(inner_item_response(ir); kwargs...)
end

function item_xs(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank})
    item_xs(inner_item_response(ir))
end

function item_ys(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank})
    item_ys(inner_item_response(ir))
end

"""
$(TYPEDSIGNATURES)

Return a view of cached log probabilities for an item. Without `response`,
the result is a matrix with rows ordered as `[false, true]` and columns for
grid points. With a Boolean response (or its integer encoding `0`/`1`),
return the corresponding vector over grid points.
"""
function item_log_ys(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank})
    return @view ir.item_bank.log_ys[:, :, ir.index]
end

function item_log_ys(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank}, response)
    return @view ir.item_bank.log_ys[Int(response) + 1, :, ir.index]
end

function resp_vec(ir::ItemResponse{<:DichotomousPointsWithLogsItemBank}, θ)
    item_bank = DichotomousSmoothedItemBank(
        ir.item_bank.inner_bank, NearestNeighborSmoother())
    resp_vec(ItemResponse(item_bank, ir.index), θ)
end

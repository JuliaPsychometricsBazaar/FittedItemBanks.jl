"""
$(TYPEDEF)

Wraps `inner`, an item bank implementing [`log_resp`](@ref) and
[`log_resp_vec`](@ref), so that [`resp`](@ref) and [`resp_vec`](@ref) return
`LogarithmicNumbers.ULogarithmic` probabilities. These retain small probabilities
that would underflow in ordinary floating-point arithmetic. Log responses are
forwarded directly to the inner bank.
"""
struct LogItemBank{ItemBankT <: AbstractItemBank} <: AbstractItemBank
    inner::ItemBankT
end

inner_item_response(ir::ItemResponse{<:LogItemBank}) = ItemResponse(ir.item_bank.inner, ir.index)

function resp(ir::ItemResponse{<:LogItemBank}, θ)
    exp(ULogarithmic, log_resp(inner_item_response(ir), θ))
end

function resp(ir::ItemResponse{<:LogItemBank}, response, θ)
    exp(ULogarithmic, log_resp(inner_item_response(ir), response, θ))
end

function resp_vec(ir::ItemResponse{<:LogItemBank}, θ)
    exp.(ULogarithmic, log_resp_vec(inner_item_response(ir), θ))
end

log_resp(ir::ItemResponse{<:LogItemBank}, θ) = log_resp(inner_item_response(ir), θ)
log_resp(ir::ItemResponse{<:LogItemBank}, response, θ) = log_resp(inner_item_response(ir), response, θ)
log_resp_vec(ir::ItemResponse{<:LogItemBank}, θ) = log_resp_vec(inner_item_response(ir), θ)

@forward LogItemBank.inner Base.length, domdims, ResponseType, DomainType

num_response_categories(ir::ItemResponse{<:LogItemBank}) = num_response_categories(inner_item_response(ir))

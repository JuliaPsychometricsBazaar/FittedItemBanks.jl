using Setfield
using Lazy: @forward

struct FixedGuessItemBank{InnerItemBank <: AbstractItemBank} <: AbstractItemBank
    guess::Float64
    inner_bank::InnerItemBank
end
y_offset(item_bank::FixedGuessItemBank, item_idx) = item_bank.guess
@forward FixedGuessItemBank.inner_bank Base.length
@forward FixedGuessItemBank.inner_bank Base.ndims
function item_params(item_bank::FixedGuessItemBank, idx)
    (;item_params(item_bank.inner_bank, idx)..., guess=item_bank.guess)
end

struct FixedSlipItemBank{InnerItemBank <: AbstractItemBank} <: AbstractItemBank
    slip::Float64
    inner_bank::InnerItemBank
end
y_offset(item_bank::FixedSlipItemBank, item_idx) = item_bank.slip
@forward FixedSlipItemBank.inner_bank Base.length
@forward FixedSlipItemBank.inner_bank Base.ndims
function item_params(item_bank::FixedSlipItemBank, idx)
    (;item_params(item_bank.inner_bank, idx)..., slip=item_bank.slip)
end

struct GuessItemBank{InnerItemBank <: AbstractItemBank} <: AbstractItemBank
    guesses::Vector{Float64}
    inner_bank::InnerItemBank
end
y_offset(item_bank::GuessItemBank, item_idx) = item_bank.guesses[item_idx]
@forward GuessItemBank.inner_bank Base.length
@forward GuessItemBank.inner_bank Base.ndims
function item_params(item_bank::GuessItemBank, idx)
    (;item_params(item_bank.inner_bank, idx)..., guess=item_bank.guesses[idx])
end

struct SlipItemBank{InnerItemBank <: AbstractItemBank} <: AbstractItemBank
    slips::Vector{Float64}
    inner_bank::InnerItemBank
end
y_offset(item_bank::SlipItemBank, item_idx) = item_bank.slips[item_idx]
@forward SlipItemBank.inner_bank Base.length
@forward SlipItemBank.inner_bank Base.ndims
function item_params(item_bank::SlipItemBank, idx)
    (;item_params(item_bank.inner_bank, idx)..., slip=item_bank.slips[idx])
end

const AnySlipItemBank = Union{SlipItemBank, FixedSlipItemBank}
const AnyGuessItemBank = Union{GuessItemBank, FixedGuessItemBank}
const AnySlipOrGuessItemBank = Union{AnySlipItemBank, AnyGuessItemBank}
const AnySlipAndGuessItemBank = Union{SlipItemBank{AnyGuessItemBank}, FixedSlipItemBank{AnyGuessItemBank}}

DomainType(item_bank::AnySlipOrGuessItemBank) = DomainType(item_bank.inner_bank)
ResponseType(item_bank::AnySlipOrGuessItemBank) = ResponseType(item_bank.inner_bank)

# Ensure we always have Slip{Guess{ItemBank}}
function FixedGuessItemBank(guess::Float64, inner_bank::AnySlipItemBank)
    @set inner_bank.inner_bank = FixedGuessItemBank(guess, inner_bank.inner_bank)
end

function GuessItemBank(guesses::Vector{Float64}, inner_bank::AnySlipItemBank)
    @set inner_bank.inner_bank = GuessItemBank(guess, inner_bank.inner_bank)
end

irf_size(guess, slip) = 1.0 - guess - slip

@inline function transform_irf_y(guess::Float64, slip::Float64, y)
    guess + irf_size(guess, slip) * y
end

@inline function transform_irf_y(ir::ItemResponse{<:GuessItemBank}, response, y)
    guess = y_offset(ir.item_bank, ir.index)
    if response
        transform_irf_y(guess, 0.0, y)
    else
        transform_irf_y(0.0, guess, y)
    end
end

@inline function transform_irf_y(ir::ItemResponse{<:SlipItemBank}, response, y)
    slip = y_offset(ir.item_bank, ir.index)
    if response
        transform_irf_y(0.0, slip, y)
    else
        transform_irf_y(slip, 0.0, y)
    end
end

function (ir::ItemResponse{<:AnySlipOrGuessItemBank})(θ)
    resp(ir, θ)
end

function resp(ir::ItemResponse{<:AnySlipOrGuessItemBank}, θ)
    resp(ir, true, θ)
end

function resp(ir::ItemResponse{<:AnySlipOrGuessItemBank}, response, θ)
    transform_irf_y(ir, response, resp(inner_item_response(ir), response, θ))
end

function resp_vec(ir::ItemResponse{<:AnySlipOrGuessItemBank}, θ)
    r = resp_vec(inner_item_response(ir), θ)
    SVector(transform_irf_y(ir, false, r[1]), transform_irf_y(ir, true, r[2]))
end

# XXX: Not getting dispatched to
function (ir::ItemResponse{<:AnySlipAndGuessItemBank})(θ)
    transform_irf_y(
        y_offset(ir.item_bank.inner_bank, ir.index),
        y_offset(ir.item_bank, ir.index),
        ItemResponse(ir.item_bank.inner_bank.inner_bank, ir.index)(θ)
    )
end

function resp(ir::ItemResponse{<:AnySlipAndGuessItemBank}, response, θ)
    transform_irf_y(
        y_offset(ir.item_bank.inner_bank, ir.index),
        y_offset(ir.item_bank, ir.index),
        resp(ItemResponse(ir.item_bank.inner_bank, ir.index), response, θ)
    )
end
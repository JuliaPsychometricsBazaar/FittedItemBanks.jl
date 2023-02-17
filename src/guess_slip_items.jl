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
Responses.ResponseType(item_bank::AnySlipOrGuessItemBank) = ResponseType(item_bank.inner_bank)

# Ensure we always have Slip{Guess{ItemBank}}
function FixedGuessItemBank(guess::Float64, inner_bank::AnySlipItemBank)
    @set inner_bank.inner_bank = FixedGuessItemBank(guess, inner_bank.inner_bank)
end

function GuessItemBank(guesses::Vector{Float64}, inner_bank::AnySlipItemBank)
    @set inner_bank.inner_bank = GuessItemBank(guess, inner_bank.inner_bank)
end

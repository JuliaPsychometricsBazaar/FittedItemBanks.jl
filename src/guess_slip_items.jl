using Setfield
using Lazy: @forward
using ConstructionBase: setproperties

## GuessAndSlipItemBank implementation

struct GuessAndSlipItemBank{
    InnerItemBankT <: AbstractItemBank,
    ParamT <: Number,
    GuessArrayT <: AbstractArray{ParamT},
    SlipArrayT <: AbstractArray{ParamT}
} <: AbstractItemBank
    guesses::GuessArrayT
    slips::SlipArrayT
    inner_bank::InnerItemBankT

    function GuessAndSlipItemBank(guesses, slips, inner_bank)
        if !(ResponseType(inner_bank) isa BooleanResponse)
            error("Guess/slip item banks can only wrap item banks with a ResponseType of BooleanResponse, not $(ResponseType(item_bank))")
        end
        param_t = eltype(guesses)
        # XXX: Could also promote two to match rather than this check
        if eltype(slips) !== param_t 
            error("Guess/slip item banks must have the same parameter type for guesses and slips")
        end
        # XXX: No need for the following, but could promote guesses/slips to the same type as inner bank?
        #=
        if eltype(inner_bank) !== param_t
            error("Guess/slip item banks must have the same parameter type for inner bank and guesses/slips")
        end
        =#
        new{typeof(inner_bank), param_t, typeof(guesses), typeof(slips)}(
            guesses, slips, inner_bank
        )
    end
end

irf_size(guess, slip) = 1.0 - guess - slip

@inline function transform_irf_y(guess, slip, y)
    guess + irf_size(guess, slip) * y
end

@inline function transform_irf_y(ir::ItemResponse{<:GuessAndSlipItemBank}, response, y)
    guess = ir.item_bank.guesses[ir.index]
    slip = ir.item_bank.slips[ir.index]
    if response
        transform_irf_y(guess, slip, y)
    else
        transform_irf_y(slip, guess, y)
    end
end

@forward GuessAndSlipItemBank.inner_bank Base.length, domdims

function subset(item_bank::GuessAndSlipItemBank, idxs)
    GuessAndSlipItemBank(
        item_bank.guesses[idxs],
        item_bank.slips[idxs],
        subset(item_bank.inner_bank, idxs)
    )
end

@views function subset_view(item_bank::GuessAndSlipItemBank, idxs)
    GuessAndSlipItemBank(
        item_bank.guesses[idxs],
        item_bank.slips[idxs],
        subset_view(item_bank.inner_bank, idxs)
    )
end

function guess_slip_indicators(item_bank::GuessAndSlipItemBank)
    return (
        item_bank.guesses isa Zeros,
        item_bank.slips isa Zeros,
    )
end

function item_params(item_bank::GuessAndSlipItemBank, idx)
    guesses_zeros, slips_zeros = guess_slip_indicators(item_bank)
    if guesses_zeros && slips_zeros
        item_params(item_bank, idx)
    elseif guesses_zeros
        (;
            item_params(item_bank.inner_bank, idx)...,
            slip = item_bank.slips[idx]
        )
    elseif slips_zeros
        (;
            item_params(item_bank.inner_bank, idx)...,
            guess = item_bank.guesses[idx]
        )
    else
        (;
            item_params(item_bank.inner_bank, idx)...,
            guess = item_bank.guesses[idx],
            slip = item_bank.slips[idx]
        )
    end
end

function convert_parameter_type(T::Type, item_bank::GuessAndSlipItemBank)
    SlipItemBank(convert(Vector{T}, item_bank.slips), convert_parameter_type(T, item_bank.inner_bank))
end

DomainType(item_bank::GuessAndSlipItemBank) = DomainType(item_bank.inner_bank)
ResponseType(item_bank::GuessAndSlipItemBank) = ResponseType(item_bank.inner_bank)
function inner_item_response(ir::ItemResponse{<: GuessAndSlipItemBank})
    ItemResponse(ir.item_bank.inner_bank, ir.index)
end
num_response_categories(ir::ItemResponse{<:GuessAndSlipItemBank}) = 2

basic_item_bank(::Type{<: GuessAndSlipItemBank{T}}) where {T} = basic_item_bank(T)
basic_item_bank(::Type{T}) where {T} = T

function replace_basic_item_bank(item_bank::GuessAndSlipItemBank, new_inner)
    setproperties(item_bank; inner_bank = replace_basic_item_bank(item_bank.inner_bank, new_inner))
end

function replace_basic_item_bank(item_bank, new_inner)
    new_inner(item_bank)
end

function resp(ir::ItemResponse{<:GuessAndSlipItemBank}, θ)
    resp(ir, true, θ)
end

function resp(ir::ItemResponse{<:GuessAndSlipItemBank}, response, θ)
    transform_irf_y(ir, response, resp(inner_item_response(ir), response, θ))
end

function resp_vec(ir::ItemResponse{<:GuessAndSlipItemBank}, θ)
    r = resp_vec(inner_item_response(ir), θ)
    SVector(transform_irf_y(ir, false, r[1]), transform_irf_y(ir, true, r[2]))
end

function item_domain(ir::ItemResponse{<:GuessAndSlipItemBank}; kwargs...)
    item_domain(inner_item_response(ir); kwargs...)
end

function maxabilresp(ir::ItemResponse{<:GuessAndSlipItemBank})
    r = maxabilresp(inner_item_response(ir))
    SVector(transform_irf_y(ir, false, r[1]), transform_irf_y(ir, true, r[2]))
end

function minabilresp(ir::ItemResponse{<:GuessAndSlipItemBank})
    r = minabilresp(inner_item_response(ir))
    SVector(transform_irf_y(ir, false, r[1]), transform_irf_y(ir, true, r[2]))
end

#=
log_resp(ir::ItemResponse{<:GuessAndSlipItemBank}, response, θ) = log(resp(ir, response, θ))
log_resp(ir::ItemResponse{<:GuessAndSlipItemBank}, θ) = log(resp(ir, θ))
log_resp_vec(ir::ItemResponse{<:GuessAndSlipItemBank}, θ) = log.(resp_vec(ir, θ))
=#

const SimilarTo2PL = Union{CdfMirtItemBank, TransferItemBank, SlopeInterceptTransferItemBank, SlopeInterceptMirtItemBank}

params_per_item(::SimilarTo2PL) = 2

function params_per_item(item_bank::GuessAndSlipItemBank{<: SimilarTo2PL})
    guesses_zeros, slips_zeros = guess_slip_indicators(item_bank)
    n = 2
    if guesses_zeros && slips_zeros
        n
    elseif guesses_zeros || slips_zeros
        n + 1
    else
        n + 2
    end
end

function spec_description(item_bank::GuessAndSlipItemBank{<: SimilarTo2PL}; level=:long)
    ppi = params_per_item(item_bank)
    return spec_description(item_bank.inner_bank; ppi, level)
end

## Psuedo-constructors

function FixedGuessItemBank(guess, inner_bank)
    nitems = length(inner_bank)
    GuessAndSlipItemBank(Fill(guess, nitems), Zeros(nitems), inner_bank)
end

function FixedSlipItemBank(slip, inner_bank)
    nitems = length(inner_bank)
    GuessAndSlipItemBank(Zeros(nitems), Fill(slip, nitems), inner_bank)
end

function GuessItemBank(guesses, inner_bank)
    nitems = length(inner_bank)
    GuessAndSlipItemBank(guesses, Zeros(nitems), inner_bank)
end

function SlipItemBank(slips, inner_bank)
    nitems = length(inner_bank)
    GuessAndSlipItemBank(Zeros(nitems), slips, inner_bank)
end

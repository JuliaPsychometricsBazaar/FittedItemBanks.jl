function FixedGuessSlipItemBank(guess::Float64, slip::Float64, item_bank)
    FixedSlipItemBank(slip, FixedGuessItemBank(guess, item_bank))
end

function GuessSlipItemBank(guesses::Vector{Float64}, slips::Vector{Float64}, item_bank)
    SlipItemBank(slips, GuessItemBank(guesses, item_bank))
end

# TODO: 1PL

"""
Convenience function to construct an item bank of the standard 2-parameter
logistic single-dimensional IRT model.
"""
function ItemBank2PL(
        difficulties,
        discriminations
)
    TransferItemBank(normal_scaled_logistic, difficulties, discriminations)
end

"""
Convenience function to construct an item bank of the standard 3-parameter
logistic single-dimensional IRT model.
"""
function ItemBank3PL(
        difficulties,
        discriminations,
        guesses
)
    GuessItemBank(guesses, ItemBank2PL(difficulties, discriminations))
end

"""
Convenience function to construct an item bank of the standard 4-parameter
logistic single-dimensional IRT model.
"""
function ItemBank4PL(
        difficulties,
        discriminations,
        guesses,
        slips
)
    SlipItemBank(slips, ItemBank3PL(difficulties, discriminations, guesses))
end

"""
Convenience function to construct an item bank of the standard 2-parameter
logistic MIRT model.
"""
function ItemBankMirt2PL(
        difficulties,
        discriminations
)
    CdfMirtItemBank(normal_scaled_logistic, difficulties, discriminations)
end

"""
Convenience function to construct an item bank of the standard 3-parameter
logistic MIRT model.
"""
function ItemBankMirt3PL(
        difficulties,
        discriminations,
        guesses
)
    GuessItemBank(guesses, ItemBankMirt2PL(difficulties, discriminations))
end

"""
Convenience function to construct an item bank of the standard 4-parameter
logistic MIRT model.
"""
function ItemBankMirt4PL(
        difficulties,
        discriminations,
        guesses,
        slips
)
    SlipItemBank(slips, ItemBankMirt3PL(difficulties, discriminations, guesses))
end

function ItemBankGPCM(
        discriminations,
        cut_points
)
    GPCMItemBank(discriminations, cut_points)
end

abstract type StdModelForm end
struct StdModel2PL <: StdModelForm end
struct StdModel3PL <: StdModelForm end
struct StdModel4PL <: StdModelForm end

struct SimpleItemBankSpec{
    StdModelT <: StdModelForm, DomainTypeT <: DomainType, ResponseTypeT <: ResponseType}
    model::StdModelT
    domain::DomainTypeT
    response::ResponseTypeT
end

function constructor(::SimpleItemBankSpec{
        StdModel2PL, OneDimContinuousDomain, BooleanResponse})
    ItemBank2PL
end
function constructor(::SimpleItemBankSpec{
        StdModel3PL, OneDimContinuousDomain, BooleanResponse})
    ItemBank3PL
end
function constructor(::SimpleItemBankSpec{
        StdModel4PL, OneDimContinuousDomain, BooleanResponse})
    ItemBank4PL
end
function constructor(::SimpleItemBankSpec{
        StdModel2PL, VectorContinuousDomain, BooleanResponse})
    ItemBankMirt2PL
end
function constructor(::SimpleItemBankSpec{
        StdModel3PL, VectorContinuousDomain, BooleanResponse})
    ItemBankMirt3PL
end
function constructor(::SimpleItemBankSpec{
        StdModel4PL, VectorContinuousDomain, BooleanResponse})
    ItemBankMirt4PL
end
function constructor(::SimpleItemBankSpec{
        StdModel2PL, D, MultinomialResponse}) where {D <: ContinuousDomain}
    ItemBankGPCM
end

function ItemBank(spec::SimpleItemBankSpec, args...; kwargs...)
    constructor(spec)(args...; kwargs...)
end

@resumable function iterate_simple_item_bank_specs()
    for model in [StdModel2PL(), StdModel3PL(), StdModel4PL()]
        for domain in [VectorContinuousDomain(), OneDimContinuousDomain()]
            #responses::Vector{ResponseType} = [BooleanResponse()]
            responses = ResponseType[BooleanResponse()]
            if model isa StdModel2PL
                push!(responses, MultinomialResponse())
            end
            for response in responses
                spec = SimpleItemBankSpec(model, domain, response)
                @yield spec
            end
        end
    end
end

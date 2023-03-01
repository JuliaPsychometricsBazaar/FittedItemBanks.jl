using Random
using FittedItemBanks
using FittedItemBanks: SimpleItemBankSpec
using FittedItemBanks.DummyData

function dummy(spec::SimpleItemBankSpec{TA, OneDimContinuousDomain, TB}) where {TA, TB}
    dummy_item_bank(
        Random.default_rng(42),
        spec,
        4
    )
end

function dummy(spec::SimpleItemBankSpec{TA, VectorContinuousDomain, TB}) where {TA, TB}
    dummy_item_bank(
        Random.default_rng(42),
        spec,
        4,
        2
    )
end

function test_item_bank(item_bank)
    rng = Random.default_rng(42)
    for idx in eachindex(item_bank)
        if DomainType(item_bank) isa OneDimContinuousDomain
            theta = randn(rng)
        else
            theta = randn(rng, domdims(item_bank))
        end
        resp = resp_vec(ItemResponse(item_bank, idx), theta)
        @test isapprox(sum(resp), 1.0)
        if ResponseType(item_bank) isa BooleanResponse
            @test length(resp) == 2
        end
    end
end

for model in [StdModel2PL(), StdModel3PL(), StdModel4PL()]
    for domain in [VectorContinuousDomain(), OneDimContinuousDomain()]
        responses::Vector{ResponseType} = [BooleanResponse()]
        if model isa StdModel2PL
            push!(responses, MultinomialResponse())
        end
        for response in responses
            item_bank = dummy(SimpleItemBankSpec(model, domain, response))
            test_item_bank(item_bank)
        end
    end
end

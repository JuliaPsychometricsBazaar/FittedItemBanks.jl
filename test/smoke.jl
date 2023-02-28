using FittedItemBanks
using FittedItemBanks.DummyData

function dummy(spec::SimpleItemBankSpec{DomainT <: OneDimContinuousDomain})
    dummy_item_bank(
        Random.default_rng(42),
        spec;
        num_questions=4
    )
end

function dummy(spec::SimpleItemBankSpec{DomainT <: VectorContinuousDomain})
    dummy_item_bank(
        Random.default_rng(42),
        spec;
        num_questions=4,
        dims=2
    )
end

function test_item_bank(item_bank)
    rng = Random.default_rng(42)
    for idx in eachindex(item_bank)
        if item_bank.domain isa OneDimContinuousDomain
            theta = rng.randn()
        else
            theta = rng.randn(domdims(item_bank))
        end
        resp = resp_vec(ItemResponse(item_bank, idx), theta)
        @test isapprox(sum(resp), 1.0)
        if item_bank.response isa BooleanResponse
            @test length(resp) == 2
        end
    end
end

for model in [StdModel2PL(), StdModel3PL(), StdModel4PL()]
    for domain in [VectorContinuousDomain(), OneDimContinuousDomain()]
        for response in [BooleanResponse(), MultinomialResponse(3)]
            item_bank = dummy(SimpleItemBankSpec(model, domain, response))
            test_item_bank(item_bank)
        end
    end
end
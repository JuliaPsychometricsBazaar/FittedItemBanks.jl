#=
Idea here is to construct dummies item bank and see that all the expected
methods match some basic invariants for single values invariants for single
values.
=#
using Random
using BSplines
using FittedItemBanks
using FittedItemBanks: SimpleItemBankSpec, iterate_simple_item_bank_specs
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

function test_bounds(lo::AbstractVector, hi, bound)
    for (l, h) in zip(lo, hi)
        test_bounds(l, h, bound)
    end
end

function test_bounds(lo, hi, bound)
    @test lo < hi
    @test -bound < lo < 0 < hi < bound
end

function test_domain(item_bank; bound = 10)
    domain = item_bank_domain(item_bank)
    @test length(domain) == 2
    test_bounds(domain[1], domain[2], bound)
end

for spec in iterate_simple_item_bank_specs()
    desc = spec_description_short(spec)
    item_bank = dummy(spec)
    @testcase "$desc" begin
        test_item_bank(item_bank)
        if !(item_bank isa NominalItemBank) # TODO
            test_domain(item_bank)
        end
    end
end

@testcase "MonopolyItemBank" begin
    item_bank = dummy_item_bank(
        Random.default_rng(42),
        MonopolyItemBank,
        4,
        3
    )
    test_item_bank(item_bank)
    test_domain(item_bank)
end

@testcase "BSplineItemBank" begin
    item_bank = dummy_item_bank(
        Random.default_rng(42),
        BSplineItemBank,
        4
    )
    test_item_bank(item_bank)
    test_domain(item_bank)
end

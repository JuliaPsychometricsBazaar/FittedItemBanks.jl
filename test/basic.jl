#=
Idea here is to construct dummies item bank and see that all the expected
methods match some basic invariants for single values invariants for single
values.
=#
using Random
using BSplines
using FittedItemBanks
using FittedItemBanks: SimpleItemBankSpec, iterate_simple_item_bank_specs, params_per_item,
                       subset, minabilresp, maxabilresp
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

function test_domain(
        item_bank; bound = 10, with_zero_symmetric = false, with_thresh = false)
    domain = item_bank_domain(item_bank)
    @test length(domain) == 2
    test_bounds(domain[1], domain[2], bound)
    if with_zero_symmetric || with_thresh
        thresh = []
        if with_thresh
            push!(thresh, :thresh => 0.2)
        end
        domain = item_bank_domain(
            item_bank; zero_symmetric = with_zero_symmetric, thresh...)
        @test length(domain) == 2
        test_bounds(domain[1], domain[2], bound)
    end
end

for spec in iterate_simple_item_bank_specs()
    desc = spec_description_short(spec)
    item_bank = dummy(spec)
    @testset "$desc" begin
        test_item_bank(item_bank)
        if !(item_bank isa NominalItemBank) && !(item_bank isa OneDimensionItemBankAdapter) # TODO
            test_domain(item_bank; with_zero_symmetric = true,
                with_thresh = params_per_item(spec.model) == 2)
        end
        @test length(subset(item_bank, [1, 3])) == 2
        ir = ItemResponse(item_bank, 1)
        idx_arg = []
        if spec.response isa MultinomialResponse
            push!(idx_arg, 1)
        end
        if spec.domain isa OneDimContinuousDomain
            @test 0 <= resp(ir, idx_arg..., 0.5) <= 1
        else
            @test 0 <= resp(ir, idx_arg..., fill(0.5, domdims(item_bank))) <= 1
        end
        if spec.domain isa OneDimContinuousDomain &&
           !(ir.item_bank isa OneDimensionItemBankAdapter)
            @test length(minabilresp(ir)) == length(maxabilresp(ir)) ==
                  num_response_categories(ir)
        end
        ppi = params_per_item(spec.model)
        num_params = length(item_params(item_bank, 1))
        ppi_desc = tryparse(Int, desc[1:1])
        if ppi_desc !== nothing
            @test ppi == ppi_desc == num_params
        end
    end
end

@testset "MonopolyItemBank" begin
    item_bank = dummy_item_bank(
        Random.default_rng(42),
        MonopolyItemBank,
        4,
        3
    )
    test_item_bank(item_bank)
    test_domain(item_bank)
end

@testset "BSplineItemBank" begin
    item_bank = dummy_item_bank(
        Random.default_rng(42),
        BSplineItemBank,
        4
    )
    test_item_bank(item_bank)
    test_domain(item_bank)
end

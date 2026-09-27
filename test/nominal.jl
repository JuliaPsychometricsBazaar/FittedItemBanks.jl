using Test
using FittedItemBanks
using FittedItemBanks: SVector

# Evaluate the model directly in higher precision, without the production
# log-density or normalization methods. BigFloat exponentials do not overflow
# or underflow at these abilities.
function nominal_probability_reference(bank, item, θ)
    setprecision(BigFloat, 256) do
        projection = sum(BigFloat.(bank.discriminations[:, item]) .* BigFloat.(θ))
        weights = exp.(BigFloat.(bank.ranks[item]) .*
            (projection .+ BigFloat.(bank.cut_points[item])))
        Float64.(weights ./ sum(weights))
    end
end

@testset "finite logits at ordinary and extreme abilities" begin
    cuts = [[-0.4, 0.6], [-0.8, 0.1, 0.9]]
    ranks = [[-1.0, 0.5], [0.0, 2.0, -0.7]]
    for discriminations in (reshape([0.7, 1.3], 1, :), [0.7 1.3; -0.2 0.4])
        for bank in (NominalItemBank(ranks, discriminations, cuts),
                GPCMItemBank(discriminations, cuts))
            for x in (-1000.0, -40.0, -0.7, 0.0, 0.9, 40.0, 1000.0)
                θ = domdims(bank) == 1 ? [x] : [x, -0.3x]
                for item in eachindex(bank)
                    ir = ItemResponse(bank, item)
                    expected = nominal_probability_reference(bank, item, θ)
                    p = resp_vec(ir, θ)
                    @test p isa SVector
                    @test all(q -> isfinite(q) && 0 <= q <= 1, p)
                    @test sum(p) ≈ 1.0
                    for (j, outcome) in enumerate(responses(ir))
                        @test isapprox(p[j], expected[j]; rtol=1e-12, atol=0)
                        @test resp(ir, outcome, θ) == p[j]
                    end
                    if domdims(bank) == 1
                        adapted_ir = ItemResponse(OneDimensionItemBankAdapter(bank), item)
                        @test resp_vec(adapted_ir, x) == p
                        for (j, outcome) in enumerate(responses(ir))
                            @test resp(adapted_ir, outcome, x) == p[j]
                        end
                    end
                end
            end
        end
    end
end

@testset "equal logits and single-category items" begin
    # These cases must retain all the mass when every unshifted exponential
    # overflows or underflows, rather than choosing a single winning category.
    bank = NominalItemBank([[1.0, 1.0, 1.0], [1.0]], [1.0, 1.0],
        [zeros(3), zeros(1)])
    for x in (-1000.0, 1000.0)
        for (item, expected) in ((1, fill(1 / 3, 3)), (2, [1.0]))
            ir = ItemResponse(bank, item)
            @test resp_vec(ir, [x]) ≈ expected
            for (j, outcome) in enumerate(responses(ir))
                @test resp(ir, outcome, [x]) ≈ expected[j]
            end
        end
    end
end

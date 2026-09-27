using Test
using FittedItemBanks
using FittedItemBanks: item_bank_xs, item_domain, item_xs, item_ys, item_log_ys

@testset "cached probabilities and forwarding" begin
    for xs in ([-2.0, 0.0, 2.0], -2.0:2.0:2.0)
        inner = DichotomousPointsItemBank(xs, [0.0 1e-20; 0.2 0.7; 1.0 0.9])
        bank = DichotomousPointsWithLogsItemBank(inner)
        @test bank.inner_bank === inner
        @test size(bank.log_ys) == (2, 3, 2)
        @test length(bank) == length(inner)
        @test eachindex(bank) == eachindex(inner)
        @test domdims(bank) == domdims(inner)
        @test DomainType(bank) == DomainType(inner)
        @test ResponseType(bank) == ResponseType(inner)
        @test item_bank_xs(bank) === xs
        smooth = DichotomousSmoothedItemBank(inner, NearestNeighborSmoother())
        for i in eachindex(bank)
            ir, inner_ir = ItemResponse(bank, i), ItemResponse(inner, i)
            @test item_xs(ir) === xs
            @test item_ys(ir) == item_ys(inner_ir)
            @test item_domain(ir) == item_domain(inner_ir)
            @test num_response_categories(ir) == 2
            logs = item_log_ys(ir)
            @test logs isa SubArray
            @test parent(logs) === bank.log_ys
            for (j, response) in enumerate(responses(ir))
                expected = response ? log.(inner.ys[:, i]) : log1p.(-inner.ys[:, i])
                @test logs[j, :] == expected
                @test item_log_ys(ir, response) == expected
                @test item_log_ys(ir, Int(response)) == expected
                @test parent(item_log_ys(ir, response)) === bank.log_ys
                @test exp.(expected) ≈ item_ys(ir, response)
            end
            for θ in (-10.0, -2.0, -1.0, 0.3, 2.0, 10.0)
                @test resp_vec(ir, θ) == resp_vec(ItemResponse(smooth, i), θ)
            end
        end
        @test item_log_ys(ItemResponse(bank, 1), true)[1] == -Inf
        @test item_log_ys(ItemResponse(bank, 1), false)[3] == -Inf
        @test item_log_ys(ItemResponse(bank, 2), false)[1] ≈ -1e-20
    end
end

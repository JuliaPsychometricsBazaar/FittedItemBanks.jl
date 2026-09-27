using Test
using FittedItemBanks
using FittedItemBanks: ULogarithmic

@testset "response and metadata forwarding" begin
    for (inner, abilities) in (
            (ItemBank2PL([-0.4, 0.8], [0.7, 1.3]), [-1000.0, 0.3, 1000.0]),
            (NominalItemBank([[0.0, 1.0], [0.0, 1.0, 2.0]], [0.7, 1.3],
                [[-0.4, 0.6], [-0.8, 0.1, 0.9]]), [[-1000.0], [0.3], [1000.0]]))
        bank = LogItemBank(inner)
        @test bank.inner === inner
        @test length(bank) == length(inner)
        @test eachindex(bank) == eachindex(inner)
        @test domdims(bank) == domdims(inner)
        @test DomainType(bank) == DomainType(inner)
        @test ResponseType(bank) == ResponseType(inner)
        for i in eachindex(bank)
            ir, inner_ir = ItemResponse(bank, i), ItemResponse(inner, i)
            @test responses(ir) == responses(inner_ir)
            @test num_response_categories(ir) == num_response_categories(inner_ir)
            for θ in abilities
                ps = resp_vec(ir, θ)
                @test eltype(ps) <: ULogarithmic
                @test log.(ps) == log_resp_vec(inner_ir, θ)
                @test log_resp_vec(ir, θ) == log_resp_vec(inner_ir, θ)
                for (j, outcome) in enumerate(responses(ir))
                    p = resp(ir, outcome, θ)
                    @test p isa ULogarithmic
                    @test p == ps[j]
                    @test log(p) == log_resp(inner_ir, outcome, θ)
                    @test log_resp(ir, outcome, θ) == log_resp(inner_ir, outcome, θ)
                end
                if ResponseType(bank) isa BooleanResponse
                    @test resp(ir, θ) == resp(ir, true, θ)
                    @test log_resp(ir, θ) == log_resp(inner_ir, θ)
                end
            end
        end
    end
end

@testset "underflow preservation" begin
    inner = ItemBank2PL([0.0], [1.0])
    ir = ItemResponse(LogItemBank(inner), 1)
    for (outcome, θ) in ((true, -1000.0), (false, 1000.0))
        @test resp(ItemResponse(inner, 1), outcome, θ) == 0.0
        p = resp(ir, outcome, θ)
        @test p > 0
        @test log(p) ≈ -1000.0
        @test log(p * p) ≈ -2000.0
    end
end

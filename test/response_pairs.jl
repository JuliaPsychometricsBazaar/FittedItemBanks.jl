using Distributions
using ForwardDiff

@testset "transfer pairs retain both tails" begin
    for T in (Float32, Float64, BigFloat), distribution in
            (Logistic(T(0), T(1)), Logistic(T(3), T(1)), Logistic(T(3), T(2)),
             Normal(T(2), T(3)), Gamma(T(2), T(3)))
        transfer = TransferItemBank(distribution, T[0], T[1])
        mirt = CdfMirtItemBank(distribution, [0.0], ones(1, 1))
        for bank in (transfer, SlopeInterceptTransferItemBank(distribution, [0.0], [1.0]),
                mirt, SlopeInterceptMirtItemBank(mirt))
            for (guess, slip) in ((T(0), T(0)), (T(0.2), T(0.1)), (T(1), T(0)), (T(0), T(1)))
                wrapped = GuessAndSlipItemBank([guess], [slip], bank)
                for x in T[-1000, -700, -40, -1, 0, 1, 3, 40, 700, 1000, -Inf, Inf]
                    theta = bank isa Union{TransferItemBank, SlopeInterceptTransferItemBank} ? x : [x]
                    ir = ItemResponse(wrapped, 1)
                    pair = resp_vec(ir, theta)
                    reference = [resp(ir, false, theta), resp(ir, true, theta)]
                    @test all(isapprox.(pair, reference; rtol=32eps(T), atol=zero(T)))
                    @test all(p -> 0 <= p <= 1, pair)
                    @test sum(pair) ≈ 1 rtol=32eps(T)
                end
            end
        end
        for x in T[-40, 0, 40]
            pair = resp_vec(ItemResponse(transfer, 1), x)
            @test eltype(pair) == T
        end
    end
end

@testset "logistic small tails and high precision reference" begin
    bank = TransferItemBank(Logistic(), [0.0], [1.0])
    ir = ItemResponse(bank, 1)
    for x in (-700.0, -40.0, 0.0, 40.0, 700.0)
        pair = resp_vec(ir, x)
        setprecision(256) do
            p = inv(1 + exp(-BigFloat(x)))
            q = inv(1 + exp(BigFloat(x)))
            @test pair[1] ≈ Float64(q) rtol=8eps() atol=0
            @test pair[2] ≈ Float64(p) rtol=8eps() atol=0
        end
    end
    @test resp_vec(ir, 40.0)[1] > 0
    @test resp_vec(ir, -40.0)[2] > 0
    @test all(isnan, resp_vec(ir, NaN))
    @test resp_vec(ir, Inf) == [0.0, 1.0]
    @test resp_vec(ir, -Inf) == [1.0, 0.0]
    @test_throws BoundsError resp_vec(ItemResponse(bank, 2), 0.0)
    for x in (-40., -0.1, 0., 0.1, 40.)
        expected = prod(resp_vec(ir, x))
        @test ForwardDiff.derivative(t -> resp_vec(ir, t)[1], x) ≈ -expected rtol=1e-12
        @test ForwardDiff.derivative(t -> resp_vec(ir, t)[2], x) ≈ expected rtol=1e-12
    end
end

function response_pair_allocations(ir, x)
    resp_vec(ir, x)
    @allocated resp_vec(ir, x)
end
@testset "ordinary pairs allocate no heap storage" begin
    bank = ItemBank4PL([0.0], [1.5], [0.1], [0.2])
    @test response_pair_allocations(ItemResponse(bank, 1), 0.7) == 0
end

function response_pairs_on_grid!(first, second, ir, grid)
    @inbounds for i in eachindex(grid)
        pair = resp_vec(ir, grid[i])
        first[i], second[i] = pair[1], pair[2]
    end
end

@testset "response loops preserve logistic tails" begin
    for T in (Float32, Float64), n in (33, 65, 129)
        grid = vcat(collect(range(T(-40), T(40); length=n)),
            T[-Inf, -745, -744, -709, 709, 744, 745, Inf, NaN])
        ir = ItemResponse(TransferItemBank(Logistic(T(0), T(1)), T[0], T[1]), 1)
        first, second = similar(grid), similar(grid)
        response_pairs_on_grid!(first, second, ir, grid)
        setprecision(256) do
            for i in eachindex(grid)
                x = BigFloat(grid[i])
                if isnan(x)
                    @test isnan(first[i]) && isnan(second[i])
                else
                    expected_first = T(inv(1 + exp(x)))
                    expected_second = T(inv(1 + exp(-x)))
                    @test first[i] ≈ expected_first rtol=32eps(T) atol=0
                    @test second[i] ≈ expected_second rtol=32eps(T) atol=0
                end
            end
        end
    end
end

@testset "logistic normalization with mixed precision" begin
    for T in (Float32, Float64, BigFloat), U in (Float32, Float64, BigFloat),
            location in (0, 3), scale in (1, 2), x in U[-40, -0.0, 0, 1, 40]
        bank = TransferItemBank(Logistic(T(location), T(scale)), T[0], T[1])
        ir = ItemResponse(bank, 1)
        pair = resp_vec(ir, x)
        reference = [resp(ir, false, x), resp(ir, true, x)]
        @test all(isapprox.(pair, reference; rtol=32max(eps(T), eps(U)), atol=0))
    end
end

@testset "unit logistic scales retain derivatives" begin
    for T in (Float32, Float64), scale in T[1, 2], x in T[-2, 0.7, 3]
        probability = s -> resp_vec(ItemResponse(
            TransferItemBank(Logistic(zero(T), s), T[0], T[1]), 1), x)[2]
        expected = -x / scale * pdf(Logistic(zero(T), scale), x)
        @test ForwardDiff.derivative(probability, scale) ≈ expected rtol=32eps(T)
    end
end

using FittedItemBanks, Test, Distributions, ForwardDiff

@testset "shared native log response pairs" begin
    for T in (Float32, Float64, BigFloat), dist in (Logistic(T(0), T(1)),
            Logistic(T(3), T(2)), Normal(T(2), T(3)), Gamma(T(2), T(3)))
        T == BigFloat && dist isa Gamma && continue # Upstream logccdf has no BigFloat method.
        banks = T == Float64 ? (TransferItemBank(dist, T[0], T[1]),
            SlopeInterceptTransferItemBank(dist, T[0], T[1])) :
            (TransferItemBank(dist, T[0], T[1]),)
        for bank in banks
            for (g,s) in ((T(0),T(0)),(T(.2),T(.1)),(T(1),T(0)),(T(0),T(1)))
                ir = ItemResponse(GuessAndSlipItemBank([g], [s], bank), 1)
                evaluate = log_resp_vec(ir)
                for x in T[-Inf,-1000,-700,-40,-1,-0.,0.,1,3,40,700,1000,Inf]
                    pair = log_resp_vec(ir, x)
                    expected = [log_resp(ir, false, x), log_resp(ir, true, x)]
                    @test all(isapprox.(pair, expected; rtol=64eps(T), atol=eps(T)))
                    @test all(isapprox.(evaluate(x), pair; rtol=64eps(T), atol=eps(T)))
                    @test eltype(evaluate(x)) == T
                    @test eltype(pair) == T
                    @test sum(exp.(pair)) ≈ one(T) rtol=64eps(T)
                end
            end
        end
    end
end

@testset "log tails and midpoint derivatives" begin
    ir = ItemResponse(TransferItemBank(Logistic(), [0.], [1.]), 1)
    for x in (-1000.,-700.,-40.,-0.1,-0.,0.,0.1,40.,700.,1000.)
        setprecision(256) do
            expected = [-log1p(exp(BigFloat(x))), -log1p(exp(-BigFloat(x)))]
            @test all(isapprox.(log_resp_vec(ir,x), Float64.(expected); rtol=16eps(), atol=0))
        end
        @test ForwardDiff.derivative(t -> log_resp_vec(ir,t)[1], x) ≈
            -resp(ir, true, x) rtol=1e-12 atol=1e-300
        @test ForwardDiff.derivative(t -> log_resp_vec(ir,t)[2], x) ≈
            resp(ir, false, x) rtol=1e-12 atol=1e-300
    end
    @test all(isnan, log_resp_vec(ir, NaN))
    @test_throws BoundsError log_resp_vec(ItemResponse(ir.item_bank,2),0.)
    for T in (Float32, Float64), U in (Float32, Float64), x in U[-40,0,40]
        mixed = ItemResponse(TransferItemBank(Logistic(T(3),T(2)), T[0],T[1]),1)
        @test log_resp_vec(mixed,x) ≈ [log_resp(mixed,false,x), log_resp(mixed,true,x)]
    end
end

@testset "MIRT log pairs share distribution dispatch" begin
    for dist in (Logistic(2., 3.), Normal(2., 3.))
        base = CdfMirtItemBank(dist, [0.], ones(1, 1))
        for bank in (base, SlopeInterceptMirtItemBank(base)), x in (-1000., -40., 0., 40., 1000.)
            ir = ItemResponse(bank, 1)
            @test log_resp_vec(ir, [x]) ≈ [log_resp(ir, false, [x]), log_resp(ir, true, [x])]
        end
    end
end

@testset "reusable log evaluators" begin
    bank = GuessAndSlipItemBank([0.2], [0.1], TransferItemBank(Logistic(), [0.], [1.]))
    ir = ItemResponse(bank,1)
    evaluate = log_resp_vec(ir)
    @test ForwardDiff.derivative(x -> evaluate(x)[1], 0.) ≈
        ForwardDiff.derivative(x -> log_resp_vec(ir,x)[1], 0.)
    bank.guesses[1] = 0.3
    @test log_resp_vec(ir)(0.) ≈ log_resp_vec(ir,0.)
    # Uncustomized banks retain the ordinary log-response dispatch.
    fallback = ItemResponse(SlopeInterceptTransferItemBank(Logistic(),[0.],[1.]),1)
    @test log_resp_vec(fallback)(0.) == log_resp_vec(fallback,0.)
end

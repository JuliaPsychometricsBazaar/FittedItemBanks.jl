using Test
using FittedItemBanks
using FittedItemBanks: ULogarithmic, Normal, Logistic, VectorOfVectors,
                       gauss_kern, uni_kern
using BSplines

@testset "response and metadata forwarding" begin
    difficulties, slopes = [-0.4, 0.8], [0.7, 1.3]
    mirt_slopes = [0.7 1.3; -0.2 0.4]
    guesses, slips = [0.1, 0.2], [0.2, 0.1]
    scalar_abilities = [-1.0, 0.3, 1.0]
    vector_abilities = [[x, -0.3x] for x in scalar_abilities]
    fixtures = []

    for distribution in (Logistic(), Normal())
        transfer = TransferItemBank(distribution, difficulties, slopes)
        mirt = CdfMirtItemBank(distribution, difficulties, mirt_slopes)
        append!(fixtures, (
            ("transfer $(typeof(distribution))", transfer, scalar_abilities),
            ("slope-intercept transfer $(typeof(distribution))",
                SlopeInterceptTransferItemBank(transfer), scalar_abilities),
            ("MIRT $(typeof(distribution))", mirt, vector_abilities),
            ("slope-intercept MIRT $(typeof(distribution))",
                SlopeInterceptMirtItemBank(mirt), vector_abilities),
            ("guess/slip transfer $(typeof(distribution))",
                GuessAndSlipItemBank(guesses, slips, transfer), scalar_abilities),
            ("guess/slip MIRT $(typeof(distribution))",
                GuessAndSlipItemBank(guesses, slips, mirt), vector_abilities),
        ))
    end

    base = ItemBank2PL(difficulties, slopes)
    append!(fixtures, (
        ("2PL", base, [-1000.0, scalar_abilities..., 1000.0]),
        ("3PL", ItemBank3PL(difficulties, slopes, guesses), scalar_abilities),
        ("4PL", ItemBank4PL(difficulties, slopes, guesses, slips), scalar_abilities),
        ("fixed guess", FixedGuessItemBank(0.2, base), scalar_abilities),
        ("fixed slip", FixedSlipItemBank(0.1, base), scalar_abilities),
        ("guess", GuessItemBank(guesses, base), scalar_abilities),
        ("slip", SlipItemBank(slips, base), scalar_abilities),
        ("certain true", FixedGuessItemBank(1.0, base), scalar_abilities),
        ("certain false", FixedSlipItemBank(1.0, base), scalar_abilities),
        ("MIRT 2PL", ItemBankMirt2PL(difficulties, mirt_slopes), vector_abilities),
        ("MIRT 3PL", ItemBankMirt3PL(difficulties, mirt_slopes, guesses), vector_abilities),
        ("MIRT 4PL", ItemBankMirt4PL(difficulties, mirt_slopes, guesses, slips), vector_abilities),
    ))

    cuts = [[-0.4, 0.6], [-0.8, 0.1, 0.9]]
    ranks = [[-1.0, 0.5], [0.0, 2.0, -0.7]]
    nominal = NominalItemBank(ranks, mirt_slopes, cuts)
    gpcm = GPCMItemBank(mirt_slopes, cuts)
    append!(fixtures, (
        ("nominal", nominal, vector_abilities),
        ("GPCM", gpcm, vector_abilities),
        ("adapted nominal", OneDimensionItemBankAdapter(
            NominalItemBank(ranks, slopes, cuts)), scalar_abilities),
        ("adapted GPCM", OneDimensionItemBankAdapter(
            GPCMItemBank(slopes, cuts)), scalar_abilities),
        ("adapted MIRT", OneDimensionItemBankAdapter(
            ItemBankMirt2PL(difficulties, reshape(slopes, 1, :))), scalar_abilities),
        ("monopoly", MonopolyItemBank(VectorOfVectors([[0.7], [1.3]]), difficulties,
            VectorOfVectors([[0.7], [1.3]])), scalar_abilities),
        ("B-spline", BSplineItemBank([BSplineBasis(4, [-2.0, 2.0]) for _ in 1:2],
            VectorOfVectors([[-2.0, -0.5, 0.5, 2.0], [-1.0, 0.0, 1.0, 3.0]])),
            scalar_abilities),
        ("nested log wrapper", LogItemBank(base), scalar_abilities),
    ))

    shared = DichotomousPointsItemBank([-2.0, 0.0, 2.0], [0.1 0.2; 0.4 0.6; 0.9 0.8])
    multi = MultiGridDichotomousPointsItemBank(
        VectorOfVectors([[-2.0, 0.0, 2.0], [-3.0, -1.0, 1.0, 3.0]]),
        VectorOfVectors([[0.1, 0.4, 0.9], [0.2, 0.3, 0.7, 0.8]]))
    for (name, points) in (("shared grid", shared), ("per-item grid", multi),
            ("cached logs", DichotomousPointsWithLogsItemBank(shared)))
        for (smoother_name, smoother, abilities) in (
                ("nearest neighbor", NearestNeighborSmoother(), scalar_abilities),
                ("Gaussian", KernelSmoother(gauss_kern, [2.0, 2.0]), scalar_abilities),
                ("uniform", KernelSmoother(uni_kern, [2.0, 2.0]), scalar_abilities))
            push!(fixtures, ("$name $smoother_name",
                DichotomousSmoothedItemBank(points, smoother), abilities))
        end
    end

    for (name, inner, abilities) in fixtures
        @testset "$name" begin
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

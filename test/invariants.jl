using Test
using FittedItemBanks
using FittedItemBanks: Normal, Logistic, VectorOfVectors, gauss_kern, uni_kern,
                       quad_kern
using BSplines

const BankTests = Base.get_extension(FittedItemBanks, :TestExt)

@testset "item bank invariants" begin
    @test BankTests !== nothing
    scalar_points = [-1000.0, -40.0, -0.7, 0.0, 0.9, 40.0, 1000.0]
    vector_points = [[x, -0.3x] for x in scalar_points]
    difficulties = [-0.4, 0.8]
    slopes = [0.7, 1.3]
    mirt_slopes = [0.7 1.3; -0.2 0.4]
    guesses, slips = [0.0, 0.2], [0.1, 0.0]
    # All concrete continuous types, plus constructor/wrapper combinations whose
    # dispatch and numerical behavior differ. Keep item parameters asymmetric.
    fixtures = []
    for distribution in (Logistic(), Normal())
        transfer = TransferItemBank(distribution, difficulties, slopes)
        mirt = CdfMirtItemBank(distribution, difficulties, mirt_slopes)
        for (name, bank, points) in (
                ("transfer", transfer, scalar_points),
                ("slope-intercept transfer", SlopeInterceptTransferItemBank(transfer), scalar_points),
                ("MIRT", mirt, vector_points),
                ("slope-intercept MIRT", SlopeInterceptMirtItemBank(mirt), vector_points))
            push!(fixtures, ("$(typeof(distribution)) $name", bank, points, true))
            push!(fixtures, ("$(typeof(distribution)) guess/slip $name",
                GuessAndSlipItemBank(guesses, slips, bank), points, true))
        end
    end
    base = ItemBank2PL(difficulties, slopes)
    push!(fixtures, ("logarithmic 2PL", LogItemBank(base), scalar_points, true))
    for (name, bank) in (
            ("2PL", base),
            ("3PL", ItemBank3PL(difficulties, slopes, guesses)),
            ("4PL", ItemBank4PL(difficulties, slopes, guesses, slips)),
            ("fixed guess", FixedGuessItemBank(0.2, base)),
            ("fixed slip", FixedSlipItemBank(0.1, base)),
            ("slip", SlipItemBank(slips, base)),
            ("zero guess/slip", GuessAndSlipItemBank(zeros(2), zeros(2), base)),
            ("certain true", FixedGuessItemBank(1.0, base)),
            ("certain false", FixedSlipItemBank(1.0, base)))
        push!(fixtures, (name, bank, scalar_points, !(name in ("certain true", "certain false"))))
    end
    for (name, bank) in (
            ("MIRT 2PL", ItemBankMirt2PL(difficulties, mirt_slopes)),
            ("MIRT 3PL", ItemBankMirt3PL(difficulties, mirt_slopes, guesses)),
            ("MIRT 4PL", ItemBankMirt4PL(difficulties, mirt_slopes, guesses, slips)))
        push!(fixtures, (name, bank, vector_points, true))
    end
    cuts = [[-0.4, 0.6], [-0.8, 0.1, 0.9]]
    ranks = [[-1.0, 0.5], [0.0, 2.0, -0.7]]
    push!(fixtures, ("nominal", NominalItemBank(ranks, mirt_slopes, cuts), vector_points, true))
    push!(fixtures, ("GPCM", GPCMItemBank(mirt_slopes, cuts), vector_points, true))
    for (name, bank) in (
            ("adapted nominal", NominalItemBank(ranks, slopes, cuts)),
            ("adapted GPCM", GPCMItemBank(slopes, cuts)),
            ("adapted MIRT", ItemBankMirt2PL(difficulties, reshape(slopes, 1, :))))
        push!(fixtures, (name, OneDimensionItemBankAdapter(bank), scalar_points, true))
    end
    # Deterministic polynomial and spline logits, with more than one item.
    monopoly = MonopolyItemBank(VectorOfVectors([[0.7], [1.3]]), difficulties,
        VectorOfVectors([[0.7], [1.3]]))
    spline = BSplineItemBank([BSplineBasis(4, [-2.0, 2.0]) for _ in 1:2],
        VectorOfVectors([[-2.0, -0.5, 0.5, 2.0], [-1.0, 0.0, 1.0, 3.0]]))
    push!(fixtures, ("monopoly", monopoly, scalar_points, true))
    push!(fixtures, ("B-spline", spline, scalar_points, true))

    shared = DichotomousPointsItemBank([-2.0, 0.0, 2.0], [0.0 0.1; 0.4 0.6; 1.0 0.9])
    with_logs = DichotomousPointsWithLogsItemBank(shared)
    multi = MultiGridDichotomousPointsItemBank(
        VectorOfVectors([[-2.0, 0.0, 2.0], [-3.0, -1.0, 1.0, 3.0]]),
        VectorOfVectors([[0.0, 0.4, 1.0], [0.1, 0.3, 0.7, 0.9]]))
    for (name, points_bank) in (("shared grid", shared), ("per-item grid", multi),
            ("cached logs", with_logs))
        @testset "$name" begin
            BankTests.test_points_item_bank(points_bank)
        end
        push!(fixtures, ("$name nearest neighbor",
            DichotomousSmoothedItemBank(points_bank, NearestNeighborSmoother()),
            scalar_points, false))
        for (kernel_name, kernel) in (("Gaussian", gauss_kern), ("uniform", uni_kern), ("quadratic", quad_kern))
            # Compact kernels require a point within kernel support. Gaussian
            # kernels have positive mass at all finite abilities mathematically.
            points = kernel === gauss_kern ? scalar_points : [-0.7, 0.0, 0.9]
            push!(fixtures, ("$name $kernel_name smoother",
                DichotomousSmoothedItemBank(points_bank, KernelSmoother(kernel, [2.0, 2.0])),
                points, true))
        end
    end
    # Guard against silently omitting a new concrete item-bank family.
    concrete_types = Set(value for name in names(FittedItemBanks; all=true)
        for value in (getfield(FittedItemBanks, name),)
        if value isa Union{DataType, UnionAll} && value <: AbstractItemBank && !isabstracttype(value))
    tested_types = Set(Base.typename(typeof(bank)).wrapper
        for bank in [shared, multi, with_logs, [fixture[2] for fixture in fixtures]...])
    @test concrete_types == tested_types
    for (name, bank, points, positive) in fixtures
        @testset "$name" begin
            BankTests.test_item_bank(bank, points; strictly_positive=positive)
        end
    end
end

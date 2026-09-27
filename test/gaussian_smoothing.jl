using Test
using FittedItemBanks
using FittedItemBanks: gauss_kern, uni_kern, quad_kern, VectorOfVectors,
                       item_xs, item_ys, log_resp, log_resp_vec

# Direct high-precision Gaussian weights provide an independent reference;
# their exponentials remain representable at the tested finite abilities.
function gaussian_response_reference(xs, ys, bandwidth, θ)
    setprecision(BigFloat, 256) do
        weights = exp.(-((BigFloat.(xs) .- BigFloat(θ)) ./ BigFloat(bandwidth)).^2 ./ 2)
        probabilities = BigFloat.(ys)
        masses = [sum(weights .* (1 .- probabilities)), sum(weights .* probabilities)]
        p = masses ./ sum(weights)
        Float64.(p), Float64.(log.(p))
    end
end

@testset "ordinary and log responses against high precision" begin
    shared = DichotomousPointsItemBank([-2.0, 0.0, 2.0],
        [0.0 0.1; 0.4 0.6; 1.0 0.9])
    multi = MultiGridDichotomousPointsItemBank(
        VectorOfVectors([[-2.0, 0.0, 2.0], [-3.0, -1.0, 1.0, 3.0]]),
        VectorOfVectors([[0.0, 0.4, 1.0], [0.1, 0.3, 0.7, 0.9]]))
    for points in (shared, multi)
        bandwidths = [0.5, 2.0]
        bank = DichotomousSmoothedItemBank(points, KernelSmoother(gauss_kern, bandwidths))
        for item in eachindex(bank), θ in (-1000.0, -40.0, -0.7, 0.0, 0.9, 40.0, 1000.0)
            ir, grid_ir = ItemResponse(bank, item), ItemResponse(points, item)
            expected, expected_logs = gaussian_response_reference(
                item_xs(grid_ir), item_ys(grid_ir), bandwidths[item], θ)
            p, lp = resp_vec(ir, θ), log_resp_vec(ir, θ)
            @test all(q -> isfinite(q) && 0 <= q <= 1, p)
            @test sum(p) ≈ 1.0
            @test p ≈ expected
            @test all(isfinite, lp)
            @test all(l -> l <= 0, lp)
            @test isapprox(lp, expected_logs; rtol=1e-12, atol=1e-12)
            @test sum(exp.(lp)) ≈ 1.0
            @test resp(ir, θ) == resp(ir, true, θ)
            @test log_resp(ir, θ) == log_resp(ir, true, θ)
            for (j, outcome) in enumerate((false, true))
                @test isapprox(resp(ir, outcome, θ), expected[j]; rtol=1e-12, atol=1e-12)
                @test log_resp(ir, outcome, θ) == lp[j]
                @test isapprox(lp[j], expected_logs[j]; rtol=1e-12, atol=1e-12)
            end
        end
    end
end

@testset "exact zeros, ones, and single-point grids" begin
    for xs in ([-2.0, 0.0, 2.0], [0.0]), y in (0.0, 0.3, 1.0)
        points = DichotomousPointsItemBank(xs, fill(y, length(xs), 1))
        bank = DichotomousSmoothedItemBank(points, KernelSmoother(gauss_kern, [0.5]))
        ir = ItemResponse(bank, 1)
        for θ in (-1000.0, 0.0, 1000.0)
            @test resp_vec(ir, θ) ≈ [1 - y, y]
            @test log_resp_vec(ir, θ) ≈ [log1p(-y), log(y)]
            @test log_resp(ir, false, θ) ≈ log1p(-y)
            @test log_resp(ir, true, θ) ≈ log(y)
        end
    end
end

@testset "compact kernels retain their support" begin
    points = DichotomousPointsItemBank([-1.0, 0.0, 1.0], reshape([0.1, 0.5, 0.9], :, 1))
    for kernel in (uni_kern, quad_kern)
        bank = DichotomousSmoothedItemBank(points, KernelSmoother(kernel, [1.0]))
        ir = ItemResponse(bank, 1)
        @test resp(ir, 0.0) ≈ 0.5
        # These kernels have genuinely zero support here; Gaussian stabilization
        # must not change their behavior or invent a nearest-neighbor fallback.
        @test isnan(resp(ir, 1000.0))
    end
end

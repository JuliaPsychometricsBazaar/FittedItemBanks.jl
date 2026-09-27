module TestExt

using Test
using FittedItemBanks
using FittedItemBanks: log_resp, log_resp_vec, item_xs, item_ys, PointsItemBank
using LogExpFunctions: logsumexp

export test_item_bank, test_item_response, test_points_item_bank

# Record absent interface methods as failures, then keep testing other invariants.
function supports(f, args...)
    available = applicable(f, args...)
    @test applicable(f, args...)
    return available
end

"""
    test_item_response(ir, points; strictly_positive=false, rtol=1e-10, atol=1e-12)

Check category ordering, probability bounds/normalization, scalar/vector agreement,
Boolean shorthand, and the corresponding log-response contract at supplied valid
abilities. Log probabilities may be `-Inf` for exact zeros, but never NaN or +Inf.
Set `strictly_positive=true` only when every category has mathematically positive
probability at every supplied point: finite logs are then required even when the
ordinary probability underflows. Include extreme *finite* abilities to distinguish
stable log implementations from `log(resp(...))`.

Ordinary probability comparisons use both `rtol` and `atol`, allowing small
absolute errors from complement subtraction near zero. Log stability is checked
separately through normalization, scalar/vector agreement, and (when requested)
finite logs. No monotonicity or common category count across items is assumed.
"""
function test_item_response(ir::ItemResponse, points;
        strictly_positive=false, rtol=1e-10, atol=1e-12)
    isempty(points) && throw(ArgumentError("provide at least one valid ability"))
    outcomes = responses(ir)
    @test outcomes isa AbstractVector
    @test !isempty(outcomes)
    @test allunique(outcomes)
    if supports(num_response_categories, ir)
        @test num_response_categories(ir) == length(outcomes)
    end
    boolean = ResponseType(ir.item_bank) isa BooleanResponse
    if boolean
        @test collect(outcomes) == [false, true]
    end
    # Test dispatch once per item, independently of ordinary probability evaluation.
    θ0 = first(points)
    has_log_vec = supports(log_resp_vec, ir, θ0)
    has_log_scalar = [supports(log_resp, ir, outcome, θ0) for outcome in outcomes]
    has_log_short = boolean && supports(log_resp, ir, θ0)
    for θ in points
        @testset "ability = $θ" begin
            @testset "probabilities" begin
                p = resp_vec(ir, θ)
                @test p isa AbstractVector
                @test length(p) == length(outcomes)
                @test all(x -> isfinite(x) && 0 <= x <= 1, p)
                @test isapprox(sum(p), 1; rtol, atol)
                for (j, outcome) in enumerate(outcomes)
                    q = resp(ir, outcome, θ)
                    @test isfinite(q) && 0 <= q <= 1
                    @test isapprox(q, p[j]; rtol, atol)
                end
                if boolean
                    @test resp(ir, θ) == resp(ir, true, θ)
                end
            end
            @testset "log probabilities" begin
                if has_log_vec
                    lp = log_resp_vec(ir, θ)
                    @test lp isa AbstractVector
                    @test length(lp) == length(outcomes)
                    @test all(x -> !isnan(x) && x <= 0, lp)
                    @test isapprox(logsumexp(lp), 0; rtol, atol)
                    if strictly_positive
                        @test all(isfinite, lp)
                    end
                    @test isapprox(exp.(lp), resp_vec(ir, θ); rtol, atol)
                    for (j, outcome) in enumerate(outcomes)
                        @test isapprox(exp(lp[j]), resp(ir, outcome, θ); rtol, atol)
                    end
                end
                for (j, outcome) in enumerate(outcomes)
                    if has_log_scalar[j]
                        lq = log_resp(ir, outcome, θ)
                        @test !isnan(lq) && lq <= 0
                        if strictly_positive
                            @test isfinite(lq)
                        end
                        @test isapprox(exp(lq), resp(ir, outcome, θ); rtol, atol)
                        if has_log_vec
                            @test isapprox(lq, log_resp_vec(ir, θ)[j]; rtol, atol)
                        end
                    end
                end
                if has_log_short
                    @test log_resp(ir, θ) == log_resp(ir, true, θ)
                end
            end
        end
    end
end

"""
    test_item_bank(bank, points; kwargs...)

Reusable continuous-bank contract checks. `points` is a nonempty collection of
valid scalar or vector abilities (vectors must match `domdims`). Keywords pass to
`test_item_response`. Scalar domains use this package's `domdims == 0` convention.
Load with `using Test, FittedItemBanks` and obtain the helper module using
`Base.get_extension(FittedItemBanks, :TestExt)`.
For raw sampled banks use `test_points_item_bank`, which tests their grid API.
"""
function test_item_bank(bank::AbstractItemBank, points; kwargs...)
    isempty(points) && throw(ArgumentError("provide at least one valid ability"))
    @testset "metadata" begin
        @test length(bank) >= 0
        @test eachindex(bank) == Base.OneTo(length(bank))
        @test DomainType(bank) isa ContinuousDomain
        @test ResponseType(bank) isa Union{BooleanResponse, MultinomialResponse}
        if supports(domdims, bank)
            if DomainType(bank) isa OneDimContinuousDomain
                @test domdims(bank) == 0
                @test all(θ -> θ isa Real, points)
            else
                @test domdims(bank) > 0
                @test all(θ -> θ isa AbstractVector && length(θ) == domdims(bank), points)
            end
        end
    end
    for i in eachindex(bank)
        @testset "item $i" begin
            test_item_response(ItemResponse(bank, i), points; kwargs...)
        end
    end
end

"""
    test_points_item_bank(bank::PointsItemBank)

Check the native sampled-data contract, without inventing interpolation or a
`resp(ir, θ)` interface for a discrete bank. Exact zero/one samples are valid.
"""
function test_points_item_bank(bank::PointsItemBank)
    @test length(bank) >= 0
    @test eachindex(bank) == Base.OneTo(length(bank))
    @test DomainType(bank) isa DiscreteDomain
    @test ResponseType(bank) isa BooleanResponse
    @test domdims(bank) == 0
    for i in eachindex(bank)
        @testset "item $i grid" begin
            ir = ItemResponse(bank, i)
            xs, ys = item_xs(ir), item_ys(ir)
            @test !isempty(xs)
            @test length(xs) == length(ys)
            @test issorted(xs) && allunique(xs)
            @test all(isfinite, xs)
            @test all(y -> isfinite(y) && 0 <= y <= 1, ys)
            @test collect(responses(ir)) == [false, true]
            if supports(num_response_categories, ir)
                @test num_response_categories(ir) == 2
            end
            @test item_ys(ir, true) == ys
            @test item_ys(ir, false) ≈ 1 .- ys
            @test item_ys(ir, false) .+ item_ys(ir, true) ≈ ones(length(xs))
        end
    end
end

end

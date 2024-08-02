"""
$(TYPEDEF)

This item bank implements the a bank with B-spline based item-responses with
dichotomous responses.

### References:

 * [*Maximum Marginal Likelihood Estimation of a
     Monotonic Polynomial Generalized Partial Credit Model*
    Winsberg, S., Thissen, D. and Wainer, H. (1984)
    ETS Research Report Series.
   ](https://doi.org/10.1002/j.2330-8516.1984.tb00080.x)
"""
struct BSplineItemBank <: AbstractItemBank
    bases::Vector{BSplineBasis{Vector{Float64}}}
    params::VectorOfVectors{Float64}
end

DomainType(::BSplineItemBank) = OneDimContinuousDomain()
ResponseType(::BSplineItemBank) = BooleanResponse()

function Base.length(item_bank::BSplineItemBank)
    length(item_bank.bases)
end

function resp_logdensity(ir::ItemResponse{<:BSplineItemBank}, θ)
    # TODO: This extrapolation does not seem to keep monotonicity :-(
    # Solvable?
    bspline_vals = bsplines_cubic_extrap(ir.item_bank.bases[ir.index], θ)
    item_params = ir.item_bank.params[ir.index]
    dot(bspline_vals, item_params)
end

function resp_vec(ir::ItemResponse{<:BSplineItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

function resp(ir::ItemResponse{<:BSplineItemBank}, outcome::Bool, θ)
    r = resp(ir, θ)
    if outcome
        r
    else
        1.0 - r
    end
end

function bsplines_unchecked(basis::BSplineBasis, leftknot, x,
        drv = NoDerivative(); derivspace = nothing)
    dest = bsplines_destarray(basis, x, drv, derivspace)
    offset = @inbounds _bsplines!(dest, derivspace, basis, x, leftknot, drv)
    bsplines_offsetarray(dest, offset, drv)
end

function bsplines_flat_extrap(basis, x)
    k = knots(basis)
    return bsplines(basis, clamp(x, k[1], k[end]))
end

function bsplines_cubic_extrap(basis, x)
    k = knots(basis)
    leftknot = intervalindex(basis, clamp(x, k[1], k[end]))
    return bsplines_unchecked(basis, leftknot, x)
end

function resp(ir::ItemResponse{<:BSplineItemBank}, θ)
    return 1.0 / (1.0 + exp(-resp_logdensity(ir, θ)))
end

function monopoly_coefficients(omega, xi, alphas, taus)
    k = length(alphas)
    λ = exp(omega)
    βs = exp.(taus)
    prev_as = zeros(2k + 1)
    as = zeros(2k + 1)
    prev_as[1] = as[1] = λ
    for i in 1:k
        for j in 1:(2i + 1)
            as[j] = prev_as[j]
            if j > 1
                as[j] += prev_as[j - 1] * -2alphas[i]
                if j > 2
                    as[j] += prev_as[j - 2] * (alphas[i]^2 + βs[i])
                end
            end
        end
        (as, prev_as) = (prev_as, as)
    end
    as = prev_as
    bs = [a / t for (a, t) in zip(as, 1:(2k + 1))]
    return (as, xi, bs)
end

"""
$(TYPEDEF)

This item bank implements the monotonic polynomial model with dichotomous responses.

```math
\\mathrm{irf}(\\theta|\\xi,{\\bf b})=\\xi+b_{1}\\theta+b_{2}\\theta^{2}+\\dots+b_{2k+1}\\theta^{2k+1}
```

```math
\\mathrm{irf}^{\\prime}(\\theta|\\mathbf{a})=a_{0}+a_{1}\\theta+a_{2}\\theta^{2}+\\cdot\\cdot\\cdot+a_{2k}\\theta^{2k}
```

### References:

 * [*Maximum Marginal Likelihood Estimation of a Monotonic Polynomial Generalized Partial Credit Model
     with Applications to Multiple Group Analysis*,
    Falk, C.F., Cai, L. (2016).
    Psychometrika.
   ](https://doi.org/10.1007/s11336-014-9428-7)
"""
struct MonopolyItemBank <: AbstractItemBank
    as::VectorOfVectorsFloat64
    xis::Vector{Float64}
    bs::VectorOfVectorsFloat64
end

DomainType(::MonopolyItemBank) = OneDimContinuousDomain()
ResponseType(::MonopolyItemBank) = BooleanResponse()

function Base.length(item_bank::MonopolyItemBank)
    length(item_bank.xis)
end

function resp_vec(ir::ItemResponse{<:MonopolyItemBank}, θ)
    resp1 = resp(ir, θ)
    SVector(1.0 - resp1, resp1)
end

function item_domain(ir::ItemResponse{<:MonopolyItemBank};
        left_mass = default_mass, right_mass = default_mass)
    right = 1.0 - right_mass
    logit(x) = log(x / (1.0 - x))
    function invert(target)
        poly = Polynomial([
            ir.item_bank.xis[ir.index] - logit(target), ir.item_bank.bs[ir.index]...])
        for root in roots(poly)
            if imag(root) == 0.0
                return real(root)
            end
        end
    end
    (
        invert(left_mass),
        invert(right)
    )
end

function resp(ir::ItemResponse{<:MonopolyItemBank}, outcome::Bool, θ)
    r = resp(ir, θ)
    if outcome
        r
    else
        1.0 - r
    end
end

function resp(ir::ItemResponse{<:MonopolyItemBank}, θ)
    m = muladd(θ, evalpoly(θ, ir.item_bank.bs[ir.index]), ir.item_bank.xis[ir.index])
    return 1.0 / (1.0 + exp(-m))
end

# TODO
#function item_domain(ir::ItemResponse{<:MonopolyItemBank}; left_mass=default_mass, right_mass=default_mass)
#end

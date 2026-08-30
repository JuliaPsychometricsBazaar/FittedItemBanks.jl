function dummy_item_bank(
        rng::AbstractRNG,
        ::Type{MonopolyItemBank},
        num_items,
        k
)
    omegas = randn(rng, num_items)
    xis = rand(rng, num_items)
    alphas = [rand(rng, k) for _ in 1:num_items]
    taus = [sort!(randn(rng, k)) for _ in 1:num_items]

    items_xi = Vector{Float64}(undef, num_items)
    items_as = VectorOfVectors{Float64}()
    items_bs = VectorOfVectors{Float64}()
    for i in 1:num_items
        (as, xi, bs) = FittedItemBanks.monopoly_coefficients(
            omegas[i], xis[i], alphas[i], taus[i])
        push!(items_as, as)
        items_xi[i] = xi
        push!(items_bs, bs)
    end

    return MonopolyItemBank(items_as, items_xi, items_bs)
end

function dummy_item_bank(
        rng::AbstractRNG,
        ::Type{BSplineItemBank},
        num_items
)
    bases = [BSplineBasis(4, [-6.0, 6.0])]
    params = VectorOfVectors{Float64}()
    for basis in bases
        push!(params, randn(rng, 4) * 20)
    end
    return BSplineItemBank(bases, params)
end

function convert_dummy_item_bank(
        ::Type{DichotomousPointsItemBank},
        item_bank,
        xs
)
    gridify(item_bank, collect(Float64, xs))
end

function convert_dummy_item_bank(
        ::Type{DichotomousSmoothedItemBank},
        item_bank,
        xs;
        bandwidth = 0.5
)
    points = convert_dummy_item_bank(DichotomousPointsItemBank, item_bank, xs)
    smoother = KernelSmoother(gauss_kern, fill(Float64(bandwidth), length(points)))
    DichotomousSmoothedItemBank(points, smoother)
end

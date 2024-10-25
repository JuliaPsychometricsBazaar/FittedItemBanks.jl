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

dummy_item_bank(type::Type, args...) = dummy_item_bank(Random.default_rng(), type, args...)

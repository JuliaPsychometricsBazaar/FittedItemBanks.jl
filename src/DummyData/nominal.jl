function dummy_item_bank(
        rng::AbstractRNG,
        ::typeof(GPCMItemBank),
        num_items,
        dims
)
    GPCMItemBank(
        dummy_discriminations(rng, dims, num_items),
        dummy_cut_points(rng, num_items)
    )
end

function dummy_item_bank(
        rng::AbstractRNG,
        ::Type{NominalItemBank},
        num_items,
        dims
)
    cut_points = dummy_cut_points(rng, num_items)
    ranks = [shuffle(rng, 1:length(cuts)) for cuts in cut_points]
    NominalItemBank(ranks, dummy_discriminations(rng, dims, num_items), cut_points)
end

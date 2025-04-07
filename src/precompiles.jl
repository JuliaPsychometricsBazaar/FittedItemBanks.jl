using PrecompileTools: @setup_workload, @compile_workload    # this is a small dependency

@setup_workload begin
    using Random: AbstractRNG, default_rng

    for spec in iterate_simple_item_bank_specs()
        args = [default_rng(42), spec, 2]
        if spec.domain isa VectorContinuousDomain
            push!(args, 2)
            x = [0.0, 0.0]
        else
            x = 0.0
        end
        @compile_workload begin
            item_bank = DummyData.dummy_item_bank(args...)
            resp_vec(ItemResponse(item_bank, 1), x)
        end
    end
    @compile_workload begin
        item_bank = DummyData.dummy_item_bank(MonopolyItemBank, 2, 2)
        resp_vec(ItemResponse(item_bank, 1), 0.0)
    end
    @compile_workload begin
        item_bank = DummyData.dummy_item_bank(BSplineItemBank, 2)
        resp_vec(ItemResponse(item_bank, 1), 0.0)
    end
end
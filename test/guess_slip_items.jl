using Distributions: Logistic
using FittedItemBanks
using FittedItemBanks: convert_parameter_type, item_params, params_per_item

@testset "Guess/slip parameter conversion" begin
    inner = TransferItemBank(Logistic(), [-0.5, 0.25], [1.0, 1.5])
    guesses = [0.2, 0.15]
    slips = [0.1, 0.05]
    banks = (
        ("guess and slip", GuessAndSlipItemBank(guesses, slips, inner)),
        ("guess only", GuessItemBank(guesses, inner)),
        ("slip only", SlipItemBank(slips, inner)),
        ("fixed guess", FixedGuessItemBank(0.2, inner)),
        ("fixed slip", FixedSlipItemBank(0.1, inner)),
    )

    for (name, bank) in banks, T in (Float32, Float64, BigFloat)
        @testset "$name to $T" begin
            converted = convert_parameter_type(T, bank)
            @test eltype(converted.guesses) === T
            @test eltype(converted.slips) === T
            @test converted.guesses == T.(bank.guesses)
            @test converted.slips == T.(bank.slips)
            @test eltype(converted.inner_bank.difficulties) === T
            @test eltype(converted.inner_bank.discriminations) === T
            @test converted.inner_bank.distribution isa Logistic{T}
            @test params_per_item(converted) == params_per_item(bank)

            atol = 8 * max(eps(Float64), eps(T))
            for idx in eachindex(bank)
                @test keys(item_params(converted, idx)) == keys(item_params(bank, idx))
                for ability in (-3.0, 0.0, 2.0), outcome in (false, true)
                    expected = resp(ItemResponse(bank, idx), outcome, ability)
                    actual = resp(ItemResponse(converted, idx), outcome, T(ability))
                    @test isapprox(actual, expected; atol, rtol = 0)
                end
            end
        end
    end
end

@testset "Typed guess/slip constructors" begin
    inner = TransferItemBank(Logistic(), [0.0, 0.5], [1.0, 1.5])
    for T in (Float32, BigFloat)
        guesses = T[0.2, 0.15]
        slips = T[0.1, 0.05]
        for (bank, expected_guesses, expected_slips) in (
            (GuessItemBank(guesses, inner), guesses, zeros(T, 2)),
            (SlipItemBank(slips, inner), zeros(T, 2), slips),
            (FixedGuessItemBank(guesses[1], inner), fill(guesses[1], 2), zeros(T, 2)),
            (FixedSlipItemBank(slips[1], inner), zeros(T, 2), fill(slips[1], 2)),
        )
            @test eltype(bank.guesses) === T
            @test eltype(bank.slips) === T
            @test bank.guesses == expected_guesses
            @test bank.slips == expected_slips
        end
    end
end

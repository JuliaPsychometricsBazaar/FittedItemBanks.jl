using Test

@testset "aqua" begin
    include("./aqua.jl")
end

@testset "jet" begin
    include("./jet.jl")
end

@testset "structs" begin
    using CheckConcreteStructs
    using FittedItemBanks: FittedItemBanks
    @test all_concrete(FittedItemBanks)
end

@testset "basic" begin
    include("./basic.jl")
end

using JuliaFormatter
using FittedItemBanks

@testset "format" begin
    dir = pkgdir(FittedItemBanks)
    @test format(dir * "/src"; overwrite = false)
    @test format(dir * "/test"; overwrite = false)
end

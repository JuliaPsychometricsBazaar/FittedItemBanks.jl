using JuliaFormatter
using FittedItemBanks

@testcase "format" begin
    dir = pkgdir(FittedItemBanks)
    @test format(dir * "/src"; overwrite = false)
    @test format(dir * "/test"; overwrite = false)
end

using JET
using FittedItemBanks

@testset "JET checks" begin
    rep = report_package(
        FittedItemBanks;
        target_modules = (
            FittedItemBanks,
        ),
        mode = :typo
    )
    @show rep
    @test length(JET.get_reports(rep)) <= 0
    #@test_broken length(JET.get_reports(rep)) == 0
end

using XUnit

@testset "aqua" begin
    include("./aqua.jl")
end

@testset "jet" begin
    include("./jet.jl")
end

@testset "basic" begin
    include("./basic.jl")
end

@testset "format" begin
    include("./format.jl")
end

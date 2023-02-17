using FittedItemBanks
using Documenter

format = Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaPsychometricsBazzar.github.io/FittedItemBanks.jl",
)

makedocs(;
    modules=[FittedItemBanks],
    authors="Frankie Robertson",
    repo="https://github.com/JuliaPsychometricsBazzar/FittedItemBanks.jl/blob/{commit}{path}#{line}",
    sitename="FittedItemBanks.jl",
    format=format,
    pages=[
        "Getting started" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazzar/FittedItemBanks.jl",
    devbranch="main",
)

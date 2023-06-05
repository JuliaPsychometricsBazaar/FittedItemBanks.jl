using FittedItemBanks
using Documenter

format = Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaPsychometricsBazaar.github.io/FittedItemBanks.jl",
)

makedocs(;
    modules=[FittedItemBanks],
    authors="Frankie Robertson",
    repo="https://github.com/JuliaPsychometricsBazaar/FittedItemBanks.jl/blob/{commit}{path}#{line}",
    sitename="FittedItemBanks.jl",
    format=format,
    pages=[
        "Home" => "index.md",
        "Reference" => ["interface.md", "parametric.md", "non_parametric.md"]
    ],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazaar/FittedItemBanks.jl",
    devbranch="main",
)

using FittedItemBanks
using Documenter
using Documenter.Remotes: GitHub

format = Documenter.HTML(
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://JuliaPsychometricsBazaar.github.io/FittedItemBanks.jl",
)

makedocs(;
    modules=[FittedItemBanks],
    authors="Frankie Robertson",
    repo = GitHub("JuliaPsychometricsBazaar", "FittedItemBanks.jl"),
    sitename="FittedItemBanks.jl",
    format=format,
    checkdocs=:public,
    pages=[
        "Home" => "index.md",
        "Reference" => ["interface.md", "parametric.md", "non_parametric.md"]
    ],
)

deploydocs(;
    repo="github.com/JuliaPsychometricsBazaar/FittedItemBanks.jl",
    devbranch="main",
)

using ImportanceSampling
using Documenter

DocMeta.setdocmeta!(ImportanceSampling, :DocTestSetup, :(using ImportanceSampling); recursive=true)

makedocs(;
    modules=[ImportanceSampling],
    authors="Šimon Soldát",
    sitename="ImportanceSampling.jl",
    format=Documenter.HTML(;
        canonical="https://soldasim.github.io/ImportanceSampling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/soldasim/ImportanceSampling.jl",
    devbranch="main",
)

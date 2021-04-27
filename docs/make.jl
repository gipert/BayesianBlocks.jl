# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using BayesianBlocks

makedocs(
    sitename = "BayesianBlocks",
    modules = [BayesianBlocks],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://gipert.github.io/BayesianBlocks.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/gipert/BayesianBlocks.jl.git",
    forcepush = true,
    push_preview = true,
)

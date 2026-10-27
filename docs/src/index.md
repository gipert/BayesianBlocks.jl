# BayesianBlocks.jl

Bayesian Blocks finds the optimal piecewise-constant representation of
one-dimensional data, following [Scargle et al.
(2013)](https://doi.org/10.1088/0004-637X/764/2/167). Given a set of events
(for instance photon arrival times or energies) or a histogram of them, it
returns the block edges that best describe the event rate: few wide blocks
where the rate is constant, narrow blocks where it changes. The number of
blocks is set by the data, through a prior on the block count.

## Installation

```julia
julia> ]add BayesianBlocks
```

## Usage

The package provides a single function, [`BayesianBlocks.partition`](@ref), which accepts a
vector of observations, a `StatsBase.Histogram` or cell edges and counts, and
returns the optimal block edges.

```julia
using BayesianBlocks, Distributions, StatsBase

data = vcat(rand(Normal(0), 1000), rand(Cauchy(5), 1000))
data = data[(data .> -5) .& (data .< 10)]

# optimal edges from the individual events
edges = BayesianBlocks.partition(data)

# or from a histogram of them
h = fit(Histogram, data, -5:0.1:10)
edges = BayesianBlocks.partition(h)

# the edges define a variable-width histogram
hb = fit(Histogram, data, edges)
```

The run time grows with the square of the number of distinct values or bins.
For large data sets, bin the data first and pass the histogram.

See the [API](@ref) page for the full description of the keyword arguments.

## Related

A header-only C++ implementation of the same algorithm is available at
[gipert/bayesian-blocks](https://github.com/gipert/bayesian-blocks).

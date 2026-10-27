# This file is a part of BayesianBlocks.jl, licensed under the MIT License (MIT).

using BayesianBlocks
using StatsBase
using DelimitedFiles
using Test

data = readdlm("data/test.dat")[:,1]

@test_throws MethodError BayesianBlocks.partition(fit(Histogram{Float64}, data), progress=true)

edges = BayesianBlocks.partition(data, progress=true)
@test edges ≈ [-3.48528, -1.87114, -1.36282, -0.677218,
              0.659105, 1.39771, 4.06582, 5.60912,
              6.17286, 7.76634, 9.91696 ] atol = 1E-04

edges = BayesianBlocks.partition(data, logfitness=:cash, logprior=:p0, p0=0.01)
@test edges ≈ [-3.48528, -1.87114, -1.36282, -0.677218,
              0.659105, 1.39771, 4.06582, 5.60912,
              6.17286, 7.76634, 9.91696 ] atol = 1E-04

edges = BayesianBlocks.partition(data, logfitness=:cash, logprior=:gamma, gamma=0.01)
@test edges ≈ [-3.48528, -2.9689, -1.74481, -1.12669,
               -0.24271, 0.492523, 1.27286, 1.59163,
               3.22158, 4.27094, 5.56663, 6.17286,
               7.76634, 9.91696] atol = 1E-04

edges = BayesianBlocks.partition(fit(Histogram, data, nbins=100, closed = :left),
                        logfitness=:cash, logprior=:p0, p0=0.01)
@test edges ≈ [-3.6, -3.0, -1.8, -1.2, -0.4, 0.4, 1.4, 3.6, 4.2, 5.6, 6.2, 7.8, 10.0]

edges = BayesianBlocks.partition(fit(Histogram, data, nbins=100, closed = :left),
                        logfitness=:cash, logprior=:gamma, gamma=0.01)
@test edges ≈ [-3.6, -3.0, -1.8, -1.2, -0.4, 0.4, 1.4, 3.6, 4.2, 5.6, 6.2, 7.8, 10.0]

# integer data must keep its full precision
t = collect(1_000_000_000:1_000_000_100)
edges = BayesianBlocks.partition(t)
@test eltype(edges) == Float64
@test edges == [1_000_000_000, 1_000_000_100]

# returned edges are histogram bin edges, so re-binning loses no events
h = fit(Histogram, data, nbins=100, closed = :left)
edges = BayesianBlocks.partition(h)
@test all(e -> any(isapprox(e), h.edges[1]), edges)
@test sum(fit(Histogram, data, edges).weights) == length(data)

# the input histogram is left untouched, also when its weights are not copied
h32 = fit(Histogram{Int32}, data, -4:0.1:10)
w = copy(h32.weights)
BayesianBlocks.partition(h32)
@test h32.weights == w

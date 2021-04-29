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
@test edges ≈ [-3.5, -3.1, -1.8, -1.2, -0.4, 0.4, 1.4, 3.6, 4.2, 5.6, 6.2, 7.8, 9.9]

edges = BayesianBlocks.partition(fit(Histogram, data, nbins=100, closed = :left),
                        logfitness=:cash, logprior=:gamma, gamma=0.01)
@test edges ≈ [-3.5, -3.1, -1.8, -1.2, -0.4, 0.4, 1.4, 3.6, 4.2, 5.6, 6.2, 7.8, 9.9]

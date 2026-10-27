# This file is a part of BayesianBlocks.jl, licensed under the MIT License (MIT).

using Test
using Aqua
using BayesianBlocks

@testset "Package BayesianBlocks" begin
    Aqua.test_all(BayesianBlocks)
    include("test.jl")
end # testset

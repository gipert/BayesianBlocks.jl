# This file is a part of BayesianBlocks.jl, licensed under the MIT License (MIT).

using StatsBase, ProgressMeter

"""
    partition(data, logfitness=:cash, logprior=:p0, progress=false, ...)

Return an array of optimal change points for a set of one-dimensional data
(observations or an histogram of them). This is the implementation of the
bayesian blocks algorithm, as outlined in [^1].

# Arguments
- `data`: numeric array or a `StatsBase.Histogram`
- `logfitness`: log of the block fitness function to be used, choose between
  [:cash]
- `logprior`: log of the prior distribution on the number of blocks to be used,
   choose between [:gamma, :p0]
- `gamma`, `p0`...: set the parameter value for the specified prior distribution
- `progress`: display a progress bar for long computations

# Example
```julia
using Distributions, StatsBase, Plots, LinearAlgebra

data = vcat(rand(Normal(0),1000),rand(Cauchy(5),1000))
data = data[(data .> -5) .& (data .< 10)]

h = fit(Histogram, data, -5:0.1:10)

# choose to use all data or an histogram of it!
hb = fit(Histogram, data, BayesianBlocks.partition(data))
hb = fit(Histogram, data, BayesianBlocks.partition(h))

plot(data, normalized=true, st=:stephist, nbins=1000)
plot!(normalize(hb), st=:step, w = 3)
```

## Performance tips
You can convert your data container to a less precise representation to improve
the performance a bit, e.g.
```julia
x::Array{Float32} = [1.1, π, (√5-1)/2]
```

[^1]: Scargle, J et al. (2012) [https://doi.org/10.1088/0004-637X/764/2/167]
"""
function partition(x::Vector{<:Real}; kwargs...)
    # take care of repeated data
    x_sorted = sort(x)
    x_unique = [x_sorted[1]]
    x_weight::Vector{Int32} = [1]
    for i in 2:length(x_sorted)
        if x_sorted[i] == x_sorted[i-1]
            x_weight[end] += 1
        else
            push!(x_unique, x_sorted[i])
            push!(x_weight, 1)
        end
    end

    partition(x_unique, x_weight; kwargs...)
end

function partition(h::Histogram{<:Integer,1}; kwargs...)
    v = collect(h.edges[1])
    x_unique = [0.5 * (v[i] + v[i+1]) for i in 1:length(v) - 1]
    x_weight::Vector{Int32} = h.weights

    # delete empty bins
    deleteat!(x_unique, findall(iszero, x_weight))
    deleteat!(x_weight, findall(iszero, x_weight))

    partition(x_unique, x_weight; kwargs...)
end

function partition(x_unique::Vector{<:Real}, x_weight::Vector{Int32};
                   logfitness::Symbol=:cash, logprior::Symbol=:p0,
                   gamma=0.01, p0=0.01, progress::Bool=false)

    # final number of data points
    N = length(x_unique)

    if logfitness == :cash
        logf = _logf_cash
    else
        error("$logfitness function not defined!")
    end

    # (log)prior on the number of blocks, a constant added once per block
    if logprior == :gamma
        # simple γ^Nblocks prior
        ncp_prior = log(gamma)
    elseif logprior == :p0
        # Note that there was a mistake in this equation in the original Scargle
        # paper (the "log" was missing). The following corrected form is taken
        # from https://arxiv.org/abs/1304.2818
        ncp_prior = log(73.53 * p0 * N^(-0.478)) - 4
    else
        error("$logprior function not defined!")
    end

    # array of (all possible) block edges
    edges = vcat(x_unique[1],
                 (x_unique[1:end-1] .+ x_unique[2:end]) ./ 2,
                 x_unique[end])

    # cumulative counts: cumw[k+1] - cumw[r] is the number of events in the
    # block spanning cells r to k
    cumw = cumsum(vcat(0, x_weight))

    # see Sec. 2.6 in [^1]
    best = Vector{Float64}(undef, N)
    last_cp = Vector{Int}(undef, N)
    A = Vector{Float64}(undef, N)

    # display progress bar for long computations, total number of steps is
    # ∑k = N(N+1)/2
    p = Progress(N * (N + 1) ÷ 2; dt = 1, enabled = progress)

    for k in 1:N
        # fitness of all configurations with the last block starting at
        # cell r (Eq. (8) in [^1])
        for r in 1:k
            A[r] = logf(cumw[k+1] - cumw[r], edges[k+1] - edges[r]) + ncp_prior +
                   (r == 1 ? 0.0 : best[r-1])
        end

        # save best configuration
        best[k], last_cp[k] = findmax(view(A, 1:k))

        update!(p, k * (k + 1) ÷ 2)
    end

    # extract changepoints by iteratively peeling off the last block
    cp = Int[]
    i = N + 1
    while i != 0
        push!(cp, i)
        i = (i == 1 ? 0 : last_cp[i-1])
    end

    return edges[reverse(cp)]
end

# Cash statistic for a block with N_k events and width T_k (Eq. (19) in [^1])
_logf_cash(N_k, T_k) = N_k * log(N_k / T_k)

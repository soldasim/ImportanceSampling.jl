
"""
Normal proposal distribution parametrized by the mean vector and the covariance matrix.
"""
mutable struct NormalProposal <: ProposalDistribution
    dist::MvNormal
end
function NormalProposal(μ, Σ)
    return NormalProposal(MvNormal(μ, Σ))
end
function NormalProposal(x_dim::Int)
    μ = zeros(x_dim)
    Σ = I(x_dim)
    return NormalProposal(MvNormal(μ, Σ))
end

x_dim(dist::NormalProposal) = length(dist.dist)

function Distributions.rand(dist::NormalProposal, count::Int)
    return rand(dist.dist, count)
end

function Distributions.logpdf(dist::NormalProposal, x::AbstractVector{<:Real})
    return logpdf(dist.dist, x)
end

"""
Decode the real-valued parameters `θ` into valid mean vector and covariance matrix.
"""
function decode_params(dist::NormalProposal, θ::AbstractVector{<:Real})
    x_dim_ = x_dim(dist)

    μ = θ[1:x_dim_]
    G = reshape(θ[x_dim_+1:end], x_dim_, x_dim_)
    Σ = G * G'
    return μ, Σ
end

function initial_params(dist::NormalProposal, count::Int)
    x_dim_ = x_dim(dist)

    # normal distribution for the mean vector
    mean_dist = MvNormal(zeros(x_dim_), I(x_dim_))

    # Wishart distribution for the covariance matrix
    n = x_dim_
    V = (1/n) * I(x_dim_)
    cov_dist = MvNormal(zeros(x_dim_), V)

    function sample_()
        μ = rand(mean_dist)
        G = rand(cov_dist, n)
        # Σ = G * G'
        return vcat(μ, vec(G))
    end

    init_params = zeros(x_dim_ + x_dim_^2, count)
    for i in 1:count
        init_params[:, i] = sample_()
    end
    return init_params
end

function loglikelihood(dist::NormalProposal, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real})
    function loglike(θ)
        μ, Σ = decode_params(dist, θ)
        d = MvNormal(μ, Σ)
        return mapreduce(t -> t[2] * logpdf(d, t[1]), +, zip(eachcol(xs), ws))
    end
end

function set_params!(dist::NormalProposal, θ::AbstractVector{<:Real})
    μ, Σ = decode_params(dist, θ)
    dist.dist = MvNormal(μ, Σ)
end

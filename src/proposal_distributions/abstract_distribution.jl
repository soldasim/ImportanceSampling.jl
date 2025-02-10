
"""
Abstract type for proposal distributions.

Each proposal distribution has to implement the following API:
- `initial_params(::ProposalDistribution, count::Int; options::ISOptions = ISOptions())`
- `loglikelihood(::ProposalDistribution, xs::AbstractMatrix{<:Real}; options::ISOptions = ISOptions())`
- `set_params!(::ProposalDistribution, θ::AbstractVector{<:Real}; options::ISOptions = ISOptions())`

The `ProposalDistribution` should be parametrized by a vector of _real-valued_ numbers.

(This means that for example the normal distribution cannot be parametrized by the individual elements
of its covariance matrix directly as that would not necesarilly result in a positive-definite covariance matrix.
Instead, suitable parameter transformations have to be implemented as a part of the functions
of the `ProposalDistribution` API, so that the `DistributionFitter` can work with real-valued parameters.)
"""
abstract type ProposalDistribution end


# --- --- Mandatory API --- ---

"""
Return some initial parameter values for the likelihood maximization.

The parameter values may be some reasonable defaults or random values.
"""
function initial_params(::ProposalDistribution, count::Int) end

"""
Sample `count` independent samples from the `ProposalDistribution` with its current parameters.
"""
function Distributions.rand(::ProposalDistribution, count::Int) end

"""
Return the log likelihood of the given data point `x` under the distribution with the current parameters.
"""
function Distributions.logpdf(::ProposalDistribution, x::AbstractVector{<:Real}) end

"""
Return a function mapping the distribution parameters `θ` to the log-pdf of the given data `xs`.
"""
function loglikelihood(::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}) end

"""
Set the parameters of the given `ProposalDistribution` to the given values `θ`.
"""
function set_params!(::ProposalDistribution, θ::AbstractVector{<:Real}) end


# --- --- Optional API --- ---

"""
Analytically compute the optimal estimate of the distribution parameters
according to the given data `xs`.

This function is a part of the _optional_ API of the `ProposalDistribution`
and may not be implemented for every distribution.
"""
function estimate_parameters!(::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}) end


# Default implementations

Distributions.rand(dist::ProposalDistribution) = Distributions.rand(dist, 1)

Distributions.pdf(dist::ProposalDistribution, x::AbstractVector{<:Real}) = exp(Distributions.logpdf(dist, x))

loglikelihood(dist::ProposalDistribution, xs::AbstractMatrix{<:Real}) =
    loglikelihood(dist, xs, ones(size(xs, 2)))

module ImportanceSampling

export ProposalDistribution
export DistributionFitter
export SamplingMethod
export ISOptions

export NormalProposal
export AnalyticalFitter, OptimizationFitter
export AMIS

export resample

using Distributions
using LinearAlgebra
using Optimization
using ProgressMeter

include("options.jl")
include("proposal_distributions/include.jl")
include("distribution_fitters/include.jl")
include("sampling_methods/include.jl")
include("resample.jl")

end # module ImportanceSampling

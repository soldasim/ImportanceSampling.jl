var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ImportanceSampling","category":"page"},{"location":"#ImportanceSampling","page":"Home","title":"ImportanceSampling","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ImportanceSampling.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ImportanceSampling]","category":"page"},{"location":"#ImportanceSampling.AMIS","page":"Home","title":"ImportanceSampling.AMIS","text":"AMIS(; kwargs...)\n\nPerforms the AMIS (adaptive multiple importance sampling) algorithm. Returns (T * N) importance samples with weights.\n\nKeywords\n\nT::Int64: Number of iterations.\nN::Int64: Number of samples in each iteration.\n\n\n\n\n\n","category":"type"},{"location":"#ImportanceSampling.AnalyticalFitter","page":"Home","title":"ImportanceSampling.AnalyticalFitter","text":"Estimates the proposal distribution parameters analytically.\n\nOnly works with proposal distributions which implement estimate_parameters!.\n\n\n\n\n\n","category":"type"},{"location":"#ImportanceSampling.NormalProposal","page":"Home","title":"ImportanceSampling.NormalProposal","text":"Normal proposal distribution parametrized by the mean vector and the covariance matrix.\n\n\n\n\n\n","category":"type"},{"location":"#ImportanceSampling.ProposalDistribution","page":"Home","title":"ImportanceSampling.ProposalDistribution","text":"Abstract type for proposal distributions.\n\nEach proposal distribution has to implement the following API:\n\ninitial_params(::ProposalDistribution, count::Int; options::ISOptions = ISOptions())\nloglikelihood(::ProposalDistribution, xs::AbstractMatrix{<:Real}; options::ISOptions = ISOptions())\nset_params!(::ProposalDistribution, θ::AbstractVector{<:Real}; options::ISOptions = ISOptions())\n\nThe ProposalDistribution should be parametrized by a vector of real-valued numbers.\n\n(This means that for example the normal distribution cannot be parametrized by the individual elements of its covariance matrix directly as that would not necesarilly result in a positive-definite covariance matrix. Instead, suitable parameter transformations have to be implemented as a part of the functions of the ProposalDistribution API, so that the DistributionFitter can work with real-valued parameters.)\n\n\n\n\n\n","category":"type"},{"location":"#ImportanceSampling.SamplingMethod-Tuple{Any, ProposalDistribution, DistributionFitter}","page":"Home","title":"ImportanceSampling.SamplingMethod","text":"Sample from the distribution given by the logpdf log_π using the given ProposalDistribution and DistributionFitter.\n\nReturn the samples (as a column-wise matrix) and their respective weights (as a vector).\n\nThe weights are not necesarilly normalized.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{ProposalDistribution, Int64}","page":"Home","title":"Base.rand","text":"Sample count independent samples from the ProposalDistribution with its current parameters.\n\n\n\n\n\n","category":"method"},{"location":"#Distributions.logpdf-Tuple{ProposalDistribution, AbstractVector{<:Real}}","page":"Home","title":"Distributions.logpdf","text":"Return the log likelihood of the given data point x under the distribution with the current parameters.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.collect_samples-Tuple{AbstractArray{<:Real, 3}, UnitRange}","page":"Home","title":"ImportanceSampling.collect_samples","text":"Return all samples from the given iterations ts.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.collect_weights-Tuple{AbstractMatrix{<:Real}, UnitRange}","page":"Home","title":"ImportanceSampling.collect_weights","text":"Return weights of the samples from the iterations ts.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.decode_params-Tuple{NormalProposal, AbstractVector{<:Real}}","page":"Home","title":"ImportanceSampling.decode_params","text":"Decode the real-valued parameters θ into valid mean vector and covariance matrix.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.estimate_parameters!-Tuple{ProposalDistribution, AbstractMatrix{<:Real}, AbstractVector{<:Real}}","page":"Home","title":"ImportanceSampling.estimate_parameters!","text":"Analytically compute the optimal estimate of the distribution parameters according to the given data xs.\n\nThis function is a part of the optional API of the ProposalDistribution and may not be implemented for every distribution.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.exp_weights-Tuple{Any}","page":"Home","title":"ImportanceSampling.exp_weights","text":"Exponentiate the given weights in a numerically stable way.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.fit_distribution!-Tuple{DistributionFitter, ProposalDistribution, AbstractMatrix{<:Real}, AbstractVector{<:Real}}","page":"Home","title":"ImportanceSampling.fit_distribution!","text":"Find the optimal parameters of the given ProposalDistribution that best fit the given data xs.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.initial_params-Tuple{ProposalDistribution, Int64}","page":"Home","title":"ImportanceSampling.initial_params","text":"Return some initial parameter values for the likelihood maximization.\n\nThe parameter values may be some reasonable defaults or random values.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.loglikelihood-Tuple{ProposalDistribution, AbstractMatrix{<:Real}, AbstractVector{<:Real}}","page":"Home","title":"ImportanceSampling.loglikelihood","text":"Return a function mapping the distribution parameters θ to the log-pdf of the given data xs.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.optimize_multistart-Tuple{Function, AbstractMatrix{<:Real}}","page":"Home","title":"ImportanceSampling.optimize_multistart","text":"Run the optimization (maximization) procedure contained within the optimize argument multiple times and return the best local optimum found this way.\n\nReturn results of all successful runs (not just the best one) if the kwargs return_all is set to true.\n\nThrows an error if all optimization runs fail.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.resample-Tuple{AbstractMatrix{<:Real}, AbstractVector{<:Real}, Int64}","page":"Home","title":"ImportanceSampling.resample","text":"xs = resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)\n\nResample count samples from the given data set xs weighted by the given weights ws with replacement to obtain a new un-weighted data set.\n\nSome data points may repeat in the resampled data set. Increasing the sample size of the initial data set may help to reduce the number of repetitions.\n\n\n\n\n\n","category":"method"},{"location":"#ImportanceSampling.set_params!-Tuple{ProposalDistribution, AbstractVector{<:Real}}","page":"Home","title":"ImportanceSampling.set_params!","text":"Set the parameters of the given ProposalDistribution to the given values θ.\n\n\n\n\n\n","category":"method"}]
}

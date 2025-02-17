
abstract type SamplingMethod end

"""
Sample from the distribution given by the logpdf `log_π` using the given `ProposalDistribution` and `DistributionFitter`.

Return the samples (as a column-wise matrix) and their respective weights (as a vector).

The weights are _not_ necesarilly normalized.
"""
function (::SamplingMethod)(log_π, ::ProposalDistribution, ::DistributionFitter;
    options::ISOptions = ISOptions(),    
) end

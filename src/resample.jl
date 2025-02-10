
"""
    xs = resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)

Resample `count` samples from the given data set `xs` weighted by the given weights `ws`
with replacement to obtain a new un-weighted data set.

Some data points may repeat in the resampled data set. Increasing the sample size
of the initial data set may help to reduce the number of repetitions.
"""
function resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)
    return hcat(wsample(eachcol(xs), ws, count; replace=true)...)
end

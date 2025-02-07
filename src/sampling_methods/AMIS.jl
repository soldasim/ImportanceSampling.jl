
"""
    AMIS(; kwargs...)

Performs the AMIS (adaptive multiple importance sampling) algorithm.
Returns `(T * N)` importance samples with weights.

# Keywords

- `T::Int64`: Number of iterations.
- `N::Int64`: Number of samples in each iteration.
"""
@kwdef struct AMIS <: SamplingMethod
    T::Int64 = 10
    N::Int64 = 20
end

function (amis::AMIS)(π, q::ProposalDistribution, fitter::DistributionFitter;
    options::ISOptions = ISOptions(),    
)
    x_dim_ = x_dim(q)
    N, T = amis.N, amis.T

    qs = [deepcopy(q) for _ in 1:T+1]
    xs = zeros(x_dim_, N, T+1)
    P = zeros(N, T+1)
    Δ = zeros(N, T+1)
    Ω = zeros(N, T+1)

    # t = 0
    xs[:, :, 1] = rand(qs[1], N)

    P[:, 1] = π.(eachcol(xs[:, :, 1]))
    Δ[:, 1] .= N .* pdf.(Ref(qs[1]), eachcol(xs[:, :, 1]))
    Ω[:, 1] .= P[:, 1] ./ (Δ[:, 1] ./ N)

    options.info && (prog = Progress(T; desc="AMIS: "))

    # t = 1,...,T
    for t in 2:T+1
        # fit the next proposal distribution based on all the samples drawn so far
        fit_distribution!(fitter, qs[t], collect_samples(xs, 1:t-1), collect_weights(Ω, 1:t-1); options)

        # draw new samples
        xs[:, :, t] = rand(qs[t], N)

        # calculate new weights
        P[:, t] = π.(eachcol(xs[:, :, t]))
        for i in 1:t
            Δ[:, t] .+= N .* pdf.(Ref(qs[i]), eachcol(xs[:, :, t]))
        end
        Ω[:, t] .= P[:, t] ./ (Δ[:, t] ./ (t * N))

        # update old weights
        for i in 1:t-1
            Δ[:, i] .+= N .* pdf.(Ref(qs[t]), eachcol(xs[:, :, i]))
            Ω[:, i] .= P[:, i] ./ (Δ[:, i] ./ (t * N))
        end

        options.info && next!(prog)
    end

    return collect_samples(xs, 2:T+1), collect_weights(Ω, 2:T+1)
end

"""
Return samples from the first `t` iterations.
"""
function collect_samples(xs::AbstractArray{<:Real, 3}, ts::UnitRange)
    x_dim_, N, _ = size(xs)
    return reshape(xs[:, :, ts], x_dim_, length(ts) * N)
end

"""
Return weights from the first `t` iterations.
"""
function collect_weights(Ω::AbstractArray{<:Real, 2}, ts::UnitRange)
    N, _ = size(Ω)
    return reshape(Ω[:, ts], length(ts) * N)
end


struct OptimizationFitter{A} <: DistributionFitter
    algorithm::A
    multistart::Int64
    parallel::Bool
    autodiff::Optimization.AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationFitter(;
    algorithm,
    multistart = 200,
    parallel = true,
    autodiff = AutoForwardDiff(),
    kwargs...
)
    return OptimizationFitter(
        algorithm,
        multistart,
        parallel,
        autodiff,
        kwargs,
    )
end

function fit_distribution!(opt::OptimizationFitter, dist::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real};
    options::ISOptions = ISOptions(),
)
    init_θs = initial_params(dist, opt.multistart)
    ll = loglikelihood(dist, xs, ws)

    optimization_function = OptimizationFunction((θ, _) -> -ll(θ), opt.autodiff)
    optimization_problem = (init_θ) -> OptimizationProblem(optimization_function, init_θ, nothing)

    function optimize(init_θ)
        params = Optimization.solve(optimization_problem(init_θ), opt.algorithm; opt.kwargs...).u
        loglike = ll(params)
        return params, loglike
    end

    θ_opt, _ = optimize_multistart(optimize, init_θs;
        opt.parallel,
        options,
    )
    set_params!(dist, θ_opt)
end

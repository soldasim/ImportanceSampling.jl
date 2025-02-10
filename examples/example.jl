using ImportanceSampling
using Distributions
using OptimizationPRIMA
using CairoMakie

function target_pdf(x)
    y = x[1] * x[2]
    ll = logpdf(Normal(1., 0.5), y)
    logp = logpdf(MvNormal(zeros(2), ones(2)), x)
    return exp(ll + logp)
end

function example()
    amis = AMIS(;
        T = 10,
        N = 100,
    )

    q = NormalProposal([0.,0.], [1.,1.])

    fitter = OptimizationFitter(;
        algorithm = NEWUOA(),
        multistart = 20,
        parallel = true,
        rhoend = 1e-4,
    )

    options = ISOptions(;
        info = true,
        debug = false,
    )

    xs, ws = amis(target_pdf, q, fitter; options)

    plot_samples(xs, ws; title="Weighted Samples") |> display

    xs_ = resample(xs, ws, 200)
    plot_samples(xs_, ones(200); title="Un-weighted Samples") |> display

    return xs, ws
end

function plot_samples(xs, ws; title=nothing)
    fig = Figure()
    ax = Axis(fig[1, 1]; title)

    contourf!(ax, -5:0.1:5, -5:0.1:5, (x1, x2) -> target_pdf([x1, x2]))
    scatter!(ax, xs[1, :], xs[2, :], color = ws, colormap = :solar)

    return fig
end

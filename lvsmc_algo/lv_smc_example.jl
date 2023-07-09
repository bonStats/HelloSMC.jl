using DifferentialEquations

# Define the deterministic model
function lotka_volterra(dx, x, p, t)
    # Model parameters.
    α, β, γ = p
    # Current state.
    x1, x2 = x
    # Evaluate differential equations.
    dx[1] = (α - β * x2) * x1 # prey
    dx[2] = (β * x1 - γ) * x2 # predator

    return nothing
end

# Set a seed for reproducibility and import plotting packages
using Random
using StatsPlots

Random.seed!(1)
x0 = [7, 5]
tspan = (0.0, 50.0)
θ_dg = [0.3, 0.04, 0.5, 1.] # The data generating θ
prob = ODEProblem(lotka_volterra, x0, tspan)

save_at = 2.0
sol = solve(prob, p = θ_dg[1:3], saveat=save_at,  verbose = false)

# Noisy observations
y = Array(sol) + 1.0 * randn(size(Array(sol))) 

plot(sol; alpha=0.3, labels = ["Prey" "Preditor"], ylim =[0.0, 24], ylabel="population")
scatter!(sol.t, y'; color=[1 2], label="")

function log_like(θ)

    α, β, γ, σ =  θ

    # Simulate Lotka-Volterra model. 
    predicted = solve(prob, p=[α, β, γ], saveat = save_at, verbose=false)

    if !SciMLBase.successful_retcode(predicted.retcode)
        # Simulation failed, return negative infinity
        return -Inf
    end

    # Add up the log likelihood for the observed data y.
    log_likelihood = 0.0
    Σ = diagm([σ^2, σ^2])
    for i in 1:length(predicted)
        like_dist = MultivariateNormal(predicted[:,i], Σ)
        log_likelihood += logpdf(like_dist, y[:,i])
    end
    return log_likelihood
end

using Distributions
# Priors on: α, β, γ, σ 
priors = [Uniform(0., 1), Uniform(0,0.1), Uniform(0, 1), Uniform(0.1, 7)]

# Function to simulate from the prior
function sim_prior(N)
    mat = zeros(N, length(priors))
    for i in 1:N
        mat[i,:] = rand.(priors)
    end
    return mat
end

# Function to evaluate the log prior
function log_prior(θ)
    # Prior distributions.
    return sum(logpdf.(priors, θ))
end

include("smc_sampler_lanneal.jl")


include("smc_sampler_lanneal_adaptive.jl")
particles_adaptive = smc_sampler_lanneal_adaptive(1000, sim_prior, log_prior, log_like)

using KernelDensity

names = ["α", "β", "γ", "σ"]
p = []
for i in 1:4
    # dens = kde(exp.(samps[:,i]))
    dens = kde(particles_adaptive[:,i])

    #x_tic = min(dens.x)
    p_l = plot(dens, title = names[i], ticks = :native)
    vline!([(θ_dg[i])], line = (:black, 2, :dash))
    push!(p,  p_l)
end
plot(p[1], p[2], p[3], p[4])
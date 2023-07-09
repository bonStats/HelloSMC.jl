using SequentialMonteCarlo
using Distributions
using DifferentialEquations
using DelimitedFiles
using Statistics
using Plots
using LinearAlgebra
using StatsBase
using StatsPlots

y_full = readdlm("lvsmc/prey_pred2.csv", ',', Float64)
n_y = size(y_full,2) - 1 # exclude initial value
ts = 2*(1.0:1.0:n_y) # times steps for data

# particle
mutable struct LVParamParticle
    α::Float64 
    β::Float64 
    γ::Float64
    σ::Float64
    LVParamParticle() = new() # initialise empty
end

Base.vec(p::LVParamParticle) = [p.α, p.β, p.γ, p.σ]

# Prior on θ = [α, β, γ, σ]
priors = product_distribution([Uniform(0., 1), Uniform(0,0.1), Uniform(0, 1), Uniform(0.1, 7)])

function log_prior(θ::Vector{Float64})
    # Prior distributions.
    return sum(logpdf.(priors, θ))
end

# Define Lotka-Volterra model.
function lotka_volterra(du, u, p, t)
    # Model parameters.
    α, β, γ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    du[1] = (α - β * y) * x # prey
    du[2] = (β * x - γ) * y # predator

    return nothing
end


# Define initial-value problem.
u0 = y_full[:,1]
tspan = (0.0, 50.0)
prob = ODEProblem(lotka_volterra, u0, tspan, [0.1, 0.1, 0.1])

y = y_full[:,2:end]
temps = 0.8 .^ (12:-1:0)
n = length(temps) - 1

function log_like(θ::Vector{Float64})
    p = θ[1:3]
    σ = θ[4]
    # Simulate Lotka-Volterra model. 
    predicted = solve(prob, Tsit5(); p=p, saveat = ts, verbose=false, alg_hints = :stiff)

    if !SciMLBase.successful_retcode(predicted.retcode)
        #println(p)
        return -Inf
    end

    # Add up the log likelihood
    log_likelihood = 0.0
    for i in eachindex(predicted)
        like_dist = MvNormal(predicted[:,i], σ^2 * I)
        log_likelihood += logpdf(like_dist, y[:,i])
    end
    return log_likelihood
end


function proposal(currθ::Vector{Float64})
    return MvNormal(currθ, 0.01^2 * I)
end

# Metropolis-Hastings Kernel with Random Walk proposal
# Posterior = prior(θ) * likelihood(θ)ᵝ
function mh(rng, currθ::Vector{Float64}, propθ::Vector{Float64}, β::Float64)
    lp_curr = logpdf(priors, currθ) + β * log_like(currθ)
    lp_prop = logpdf(priors, propθ) + β * log_like(propθ)
    if lp_prop - lp_curr > log(rand(rng))
        return propθ
    else
        return currθ
    end
end


# mutation kernel
function M!(new::LVParamParticle, rng, t::Int64, old::LVParamParticle, ::Nothing)
    if t == 1
        θ = rand(rng, priors)
    else
        # proposal
        propθ = rand(rng, proposal(vec(old)))
        θ = mh(rng, vec(old), propθ, temps[t])
    end

    new.α, new.β, new.γ, new.σ = θ

end

# potential function
function lG(t::Int64, particle::LVParamParticle, ::Nothing)
    β_incr = temps[t+1] - temps[t]
    return β_incr * log_like(vec(particle))
end

N = 2^10        # number of particles
threads = 1     # number of threads
κ = 0.5         # relative ESS threshold
saveall = true  # save full trajectory 
model = SMCModel(M!, lG, n, LVParamParticle, Nothing)
smcio = SMCIO{model.particle, model.pScratch}(N, n, threads, saveall, κ)
smc!(model, smcio)

# η̂ = prior * likelhood
ps = vec.(smcio.zetas)
psm = Matrix(hcat(ps...)')
ws = Weights(smcio.ws)

# mean and cov
μ = mean(ps, ws)
Σ = cov(psm, ws)

# estimate variance of test function = var(σ)
SequentialMonteCarlo.V(smcio, p -> (p.σ - μ[4])^2, true, true, n)

# kde
density(psm[:,1], weights = ws)
density(psm[:,2], weights = ws)
density(psm[:,3], weights = ws)
density(psm[:,4], weights = ws)

ps = [Matrix(hcat(vec.(zs)...)') for zs in smcio.allZetas]
ws = Weights.(smcio.allWs)
Σs = cov.(ps, ws)

function proposal(currθ::Vector{Float64}, t::Int64)
    return MvNormal(currθ, (0.2^2)*Σs[t])
end

# mutation kernel
function M2!(new::LVParamParticle, rng, t::Int64, old::LVParamParticle, ::Nothing)
    if t == 1
        θ = rand(rng, priors)
    else
        # proposal
        θ = vec(old)
        for i in 1:5
            propθ = rand(rng, proposal(θ, t))
            θ = mh(rng, θ, propθ, temps[t])
        end
    end

    new.α, new.β, new.γ, new.σ = θ

end

model2 = SMCModel(M2!, lG, n, LVParamParticle, Nothing)
smcio2 = SMCIO{model.particle, model.pScratch}(N, n, threads, saveall, κ)
smc!(model2, smcio2)

# η̂ = prior * likelhood
ps2 = Matrix(hcat(vec.(smcio2.zetas)...)')
ws2 = Weights(smcio2.ws)

# mean and cov
μ2 = mean(ps2, ws2, dims = 1)
Σ2 = cov(ps2, ws2)

# unique trajectories
unique(smcio2.allEves[end])

# kde
density(ps2[:,1], weights = ws2)
density(ps2[:,2], weights = ws2)
density(ps2[:,3], weights = ws2)
density(ps2[:,4], weights = ws2)

# estimate variance of test function = var(σ)
SequentialMonteCarlo.V(smcio2, p-> (p.σ - μ[4])^2, true, true, n)

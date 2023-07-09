using Distributions, StatsBase, LinearAlgebra

function logsumexp(x)
    my_max = maximum(x)
    x_new = x .- my_max
    return my_max + log(sum(exp.(x_new)))
end

function compute_ess_diff(γ, γ_old, log_like)
    N = length(log_like)
    log_propto_W = (γ - γ_old) .* log_like
    log_propto_W = log_propto_W .- logsumexp(log_propto_W)
    return exp(-logsumexp(2.0 * log_propto_W)) - N /2
end

using Roots

function smc_sampler_lanneal_adaptive(N, simprior, logprior, loglike; Σ = nothing, R=20)

    # Sample initial from prior and set weights
    ξ = simprior(N)
    log_W = -log(N)*ones(N)

    d = size(ξ, 2) # Get dimension of θ

    # If RW sigma is not specified use a default
    if isnothing(Σ)
        Σ = cov(ξ)
    end

    loglike_curr = fill(NaN, N)
    logprior_curr = fill(NaN, N)
    for i in 1:N
        loglike_curr[i] = loglike(ξ[i, :])
        logprior_curr[i] = logprior(ξ[i, :])
    end

    γ = 0.0
  
    while γ < 1.0

        γ_old = γ
        if compute_ess_diff(1.0, γ_old, loglike_curr) > 0
            γ = 1.0
        else
            fn = x -> compute_ess_diff(x, γ_old, loglike_curr)
            γ = find_zero(fn, (γ_old, 1.0))
        end

        # Calcuate (log) weights w and normalised weights W
        log_w = log_W .+ (γ - γ_old) .* loglike_curr
        log_W .= log_w .- logsumexp(log_w) 

        # Resample the particles and reset the weights to 1/N
        inds = sample(1:N, Weights(exp.(log_W)), N) # Multinomial

        ξ = ξ[inds, :]
        loglike_curr = loglike_curr[inds]
        logprior_curr = logprior_curr[inds]

        log_W = -log(N)*ones(N)
        
        # Adaptive choice of MCMC proposal covariance
        Σ = cov(ξ)
    
        # MCMC Diversify step
        for i in 1:N
            for k in 1:R
                ξ_prop = rand(MvNormal(ξ[i, :], Σ))
                loglike_prop = loglike(ξ_prop)
                logprior_prop = logprior(ξ_prop)
                log_rmh = γ * (loglike_prop - loglike_curr[i]) + logprior_prop - logprior_curr[i]

                # Accept the proposal with probability RMH
                if rand() < exp(log_rmh)
                    ξ[i, :] = ξ_prop
                    loglike_curr[i] = loglike_prop
                    logprior_curr[i] = logprior_prop
                end
            end
        end
        println("Current γ = ",γ)
        println("Unique particles: ",length(unique(ξ[:, 1])))
    end
    return ξ
end
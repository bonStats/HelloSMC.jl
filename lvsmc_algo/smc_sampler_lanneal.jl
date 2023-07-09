using Distributions, StatsBase, LinearAlgebra

function logsumexp(x)
    my_max = maximum(x)
    x_new = x .- my_max
    return my_max + log(sum(exp.(x_new)))
end

function smc_sampler_lanneal(N, simprior, logprior, loglike; γ = range(0, stop=1, length=15).^5, Σ = nothing, R=20)

    # Sample initial from prior and set weights
    ξ = simprior(N)
    log_W = ones(N) ./ N

    d = size(ξ, 2) # Get dimension of θ

    # If RW sigma is not specified use a default
    if isnothing(Σ)
        Σ = diagm(ones(d))/R
    end

    loglike_curr = fill(NaN, N)
    logprior_curr = fill(NaN, N)
    for i in 1:N
        loglike_curr[i] = loglike(ξ[i, :])
        logprior_curr[i] = logprior(ξ[i, :])
    end
  
    for t in 2:length(γ)

        # Calcuate (log) weights w and normalised weights W
        log_w = log_W .+ (γ[t] - γ[t-1]) .* loglike_curr
        log_W .= log_w .- logsumexp(log_w) 
        
        Σ = cov(ξ) # Adaptive choice 

        # Resample the particles
        inds = sample(1:N, Weights(exp.(log_W)), N) # Multinomial

        ξ = ξ[inds, :]
        loglike_curr = loglike_curr[inds]
        logprior_curr = logprior_curr[inds]
    
        # MCMC Diversify step
        for i in 1:N
            for k in 1:R
                ξ_prop = rand(MvNormal(ξ[i, :], Σ))
                loglike_prop = loglike(ξ_prop)
                logprior_prop = logprior(ξ_prop)
                log_rmh = γ[t] * (loglike_prop - loglike_curr[i]) + logprior_prop - logprior_curr[i]

                # Accept the proposal with probability RMH
                if rand() < exp(log_rmh)
                    ξ[i, :] = ξ_prop
                    loglike_curr[i] = loglike_prop
                    logprior_curr[i] = logprior_prop
                end
            end
        end
        println("Current γ = ",γ[t])
        println("Unique particles: ",length(unique(ξ[:, 1])))
    end
    return ξ
end
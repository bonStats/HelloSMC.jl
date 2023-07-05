using SequentialMonteCarlo
using Distributions
using Catalyst
using JumpProcesses
using DelimitedFiles
using Statistics
using Plots

y = readdlm("examples/prey_pred.csv", ',', Int64)
n = size(y,2)
ts = 1.0:1.0:50.0

# particle
mutable struct LVParticle
    prey::Int64
    pred::Int64
    LVParticle() = new() # initialise empty
end

# setup MJP for use in mutation kernel
lveqns = @reaction_network begin
    c1, prey --> 2*prey         # prey reproduction
    c2, prey + pred --> 2*pred  # prey death, pred reproduction
    c3, pred --> 0              # pred death
end

rates = (:c1 => 0.5, :c2 => 0.0025, :c3 => 0.3)
initial = [71, 79] # latent at time 0.0

# mutation kernel
function M!(newParticle::LVParticle, rng, t::Int64, oldParticle::LVParticle, ::Nothing)
    if t == 1
        u0 = initial
        tspan = (0.0, ts[1])
    else
        u0 = [oldParticle.prey, oldParticle.pred]
        tspan = (ts[t-1], ts[t])
    end
    
    dprobt = DiscreteProblem(lveqns, u0, tspan, rates)
    jprobt = JumpProblem(lveqns, dprobt, Direct(), rng = rng)
    res = solve(jprobt, SSAStepper())

    newParticle.prey = res(ts[t])[1]
    newParticle.pred = res(ts[t])[2]
end

# potential function
function lG(t::Int64, particle::LVParticle, ::Nothing)
    preydist = Poisson(0.5*particle.prey)
    preddist = Poisson(0.8*particle.pred)
    return logpdf(preydist, y[1,t]) + logpdf(preddist, y[2,t])
end

N = 2^12
threads = 1
κ = 0.5 # relative ESS threshold
saveall = true
model = SMCModel(M!, lG, n, LVParticle, Nothing)
smcio = SMCIO{model.particle, model.pScratch}(N, n, threads, saveall, κ)
smc!(model, smcio)

Statistics.mean(ps::Vector{LVParticle}) = [mean(getfield.(ps, :prey)), mean(getfield.(ps, :pred))]
mean(smcio.zetas)


# load true latent states
x = readdlm("examples/prey_pred_true_latent.csv", ',', Int64)

plot(y[1,:], label="obs", title = "Prey estimated latent states vs observations")
pids = sample(1:smcio.N, 100)
for i in pids
    anc = getindex.(smcio.allEves, i)
    u = getindex.(smcio.allZetas, reverse(anc))
    plot!(getfield.(u, :prey), color = "grey", alpha = 0.5, label = "")
end
plot!(x[1,:], label="latent", color = "red")

plot(y[2,:], label="obs", title = "Pred. estimated latent states vs observations")
pids = sample(1:smcio.N, 100)
for i in pids
    anc = getindex.(smcio.allEves, i)
    u = getindex.(smcio.allZetas, reverse(anc))
    plot!(getfield.(u, :pred), color = "grey", alpha = 0.5, label = "")
end
plot!(x[2,:], label="latent", color = "red")




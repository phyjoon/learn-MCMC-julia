using Distributions
using Plots, StatsPlots

function jackknife(data, w)
    len = length(data)
    bin_num = len % w == 0 ? (len ÷ w) : (len ÷ w) + 1
    append!(data, zeros(bin_num * w - len))

    bin = zeros(bin_num)
    for i in 1:bin_num
        bin[i] = sum(data[1 + (i-1)*w : i * w]) / w
    end

    resize!(data, len)

    avg_f = sum(bin) / bin_num
    error = sqrt(sum((bin .- avg_f) .^ 2) / bin_num / (bin_num - 1))

    return [avg_f, error]
end

function action(x)
    return -x^2/2 + x^4/24
end

function action_hamiltonian(x, p)
    return action(x) + p^2/2
end

function deriv_hamiltonian(x, p)
    return -x + x^3/6, p
end

function leap_frog(x0, p0, Δτ, Nτ)
    x, p = zeros(Nτ), zeros(Nτ)

    # Prepare the iteration
    x[1] = x0 + p0 * Δτ/2           # τ = Δτ/2
    p[1] = p0                       # τ = 0

    # Iterate for (Nτ - 1) steps
    for i in 2:Nτ
        p[i] = p[i-1] - deriv_hamiltonian(x[i], p[i])[1] * Δτ   # τ = (i-1)*(Δτ)
        x[i] = x[i-1] + deriv_hamiltonian(x[i], p[i])[2] * Δτ   # τ = (i-1/2)*(Δτ)
    end

    # Final results
    pf = p[Nτ] - deriv_hamiltonian(x[Nτ], p[Nτ])[1] * Δτ        # τ = Nτ*(Δτ)
    xf = x[Nτ] + deriv_hamiltonian(x[Nτ], p[Nτ])[2] * Δτ/2      # τ = Nτ*(Δτ)

    return xf, pf
end

function hybrid_monte_carlo(sample_size, Δτ, Nτ)
    # Initial Configuration
    x = zeros(sample_size + 1)
    x[1] = 0
    idx_accept = 0

    # Main Loop
    for i in 2:(sample_size + 1)

        p = rand(Normal(0, 1))
        H_initial = action_hamiltonian(x[i-1], p)

        xf, pf = leap_frog(x[i-1], p, Δτ, Nτ)
        H_later = action_hamiltonian(xf, pf)

        dS = H_later - H_initial

        if rand(Uniform(0, 1)) < exp(-dS)
            idx_accept += 1
            x[i] = xf
        else
            x[i] = x[i-1]
        end
    end

    return x, idx_accept
end


function metropolis_hastings(step_size, sample_size)
    # Initial Configuration
    x = zeros(sample_size + 1)
    x[1] = 0
    idx_accept = 0

    # Main Loop
    for i in 1:sample_size
        x_prime = x[i] + rand(Uniform(-step_size, step_size))
        dS = action(x_prime) - action(x[i])

        if rand(Uniform(0, 1)) < exp(-dS)
            idx_accept += 1
            x[i+1] = x_prime
        else
            x[i+1] = x[i]
        end
    end

    return x, idx_accept
end

function plt(x)
    return exp(-action(x)) / 18.0006
end

# Convergence of the pdf
sampling, num_accept = metropolis_hastings(sqrt(3), 10^7)
histogram(sampling, nbins = 1000, normalize=true)
p = Plots.plot!(plt, xlims=(-5,5))
display(p)
println(num_accept)

# Convergence of expectation values
α = accumulate(+, sampling, dims=1) ./ collect(1:10^7+1)
α2 = accumulate(+, sampling .^ 2, dims=1) ./ collect(1:10^7+1)
Plots.plot(α, xaxis=:log)
p = plot!(α2, xaxis=:log)
display(p)

# Sampling error (Jackknife)
res = zeros((100, 2))
for i in 1:100
    res[i,:] = jackknife(sampling .^ 2, i)
end
p = Plots.plot(res[:,1], yerror=res[:,2])
display(p)


# Convergence of the pdf
sampling2, num_accept = hybrid_monte_carlo(10^7, 0.01, 40)
histogram(sampling2, nbins = 1000, normalize=true)
p = Plots.plot!(plt, xlims=(-5,5))
display(p)
println(num_accept)

# Convergence of expectation values
α = accumulate(+, sampling2, dims=1) ./ collect(1:10^7+1)
α2 = accumulate(+, sampling2 .^ 2, dims=1) ./ collect(1:10^7+1)
Plots.plot(α, xaxis=:log)
p = plot!(α2, xaxis=:log)
display(p)

# Sampling error (Jackknife)
res = zeros((100, 2))
for i in 1:100
    res[i,:] = jackknife(sampling2 .^ 2, i)
end
p = Plots.plot(res[:,1], yerror=res[:,2])
display(p)

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
    return x^2/2
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

# Convergence of the pdf
sampling, num_accept = metropolis_hastings(sqrt(3), 10^7)
histogram(sampling, nbins = 1000, normalize=true)
p = Plots.plot!(Normal(0,1), xlims=(-5,5))
display(p)

# Convergence of expectation values
α = accumulate(+, sampling, dims=1) ./ collect(1:10^7+1)
α2 = accumulate(+, sampling .^ 2, dims=1) ./ collect(1:10^7+1)
Plots.plot(α, xaxis=:log)
p = plot!(α2, xaxis=:log)
display(p)

# Sampling error (Jackknife)
sampling, num_accept = metropolis_hastings(sqrt(3), 5000000)
res = zeros((100, 2))
for i in 1:100
    res[i,:] = jackknife(sampling .^ 2, i)
end
Plots.plot(res[:,1], yerror=res[:,2])

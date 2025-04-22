### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 05e9d630-1f9e-11f0-2374-bdcd5b94dc21
begin
	import Pkg; Pkg.activate()

	using CairoMakie, ColorSchemes, DataFrames, Random, PlutoUI, Distributions, PlutoTeachingTools
end

# ╔═╡ 44233307-a3df-4bc8-a1cd-e99b61999950
begin
if isfile("pics/MetropolisHastings.png")
	TwoColumnWideLeft(tip(md"In statistics and statistical physics, the Metropolis–Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult. New samples are added to the sequence in two steps: first a new sample is proposed based on the previous sample, then the proposed sample is either added to the sequence or rejected depending on the value of the probability distribution at that point."), RobustLocalResource("","pics/MetropolisHastings.png"))
else
	tip(md"In statistics and statistical physics, the Metropolis–Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random samples from a probability distribution from which direct sampling is difficult. New samples are added to the sequence in two steps: first a new sample is proposed based on the previous sample, then the proposed sample is either added to the sequence or rejected depending on the value of the probability distribution at that point.")
end
end

# ╔═╡ 24dcb6f1-25cc-4175-8d9f-264cebe86008
md"By Jaewook LeeWoosuk SungJoo-Ho Choi - Metamodel for Efficient Estimation of Capacity-Fade Uncertainty in Li-Ion Batteries for Electric VehiclesJune 2015Energies 8(6):5538-5554DOI:10.3390/en8065538, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=130401255"

# ╔═╡ aa1503f6-ea8d-43f8-96a4-e53ab51ada11
"""
This is the target distribution, it can be any function so long as it satesfies a few properties:

* it should be non-negative after exponentiation.

* up to a constant, i.e. proportial PDF is fine.

* defined wherever you might propose, i.e. bounded
"""
function some_target(x)
	return  0.5 / (1 + x^2)
end

# ╔═╡ 517aa4e2-137d-4d2d-8f17-1b98a8aa81e7
md"""

The initial guess (initial_x) should be sampled from the prior
"""

# ╔═╡ d8ef1b09-ebfe-4142-b3b4-39b9edfdb38a
function run_mcmc_trace(target_prob, initial_x::Float64, n_steps::Int; proposal_std::Float64=0.5)
    x = initial_x #starting point (init state of chain)
    trace = zeros(n_steps) #sample storage
    accepted = Vector{Bool}(undef, n_steps) #
    
    for i in 1:n_steps
        x_prop = x + randn() * proposal_std 
        α = target_prob(x_prop) / target_prob(x)
		if rand() < α
		    x = x_prop
			accepted[i] = true
		else
			accepted[i] = false
		end
        trace[i] = x
    end

    return trace, accepted
end

# ╔═╡ a30d593e-5ba3-45c6-9ce2-d9f67903a485
#gen samples
samples, _= run_mcmc_trace(some_target, 0.0, 1000000, proposal_std=0.1)

# ╔═╡ 2d466e44-d3ab-491a-92d6-c11d3cb660ca
begin
	local fig = Figure(size=(800, 400))
	local ax = Axis(fig[1, 1]; xlabel="x", ylabel="Density", title="Metropolis-Hastings Sampling")

	density!(ax, samples[1000:end])#; bins=50, normalization=:pdf, color=:gray80)


	local xs = LinRange(-4, 4, 300)
	lines!(ax, xs, pdf.(Cauchy(0, 1), xs); linewidth=2, label="True PDF")

	axislegend(ax)

	xlims!(-15, 15)

	fig
end

# ╔═╡ f2f3b4e5-aac4-491b-a47b-3d1fadb08b33
begin
	num_samples = 4000
	trace, accepted = run_mcmc_trace(some_target, 0.0, num_samples)
end

# ╔═╡ ab11cf92-a81f-4ee7-9dc4-b13922bec273
@bind step PlutoUI.Slider(1:length(trace); show_value=true)

# ╔═╡ 12b931b8-7818-48bb-a1ab-4fbd4e44cb43
begin
	local fig = Figure(size=(800, 400))
	local ax = Axis(fig[1, 1], xlabel="x", ylabel="Density", title="Growing MCMC Histogram")
	
	# Plot histogram of current samples
	if step ≥ 3
	    #hist!(ax, samples[1:step]; bins=50, normalization=:pdf, color=:skyblue, strokewidth=0.5)
		hist!(ax, samples[1:step]; bins=-15.0:0.2:15.0, normalization=:pdf, color=:skyblue, strokewidth=0.5)
	else
	    vlines!(ax, samples[1:step], [0.0]; color=:skyblue, linewidth=5)
	end
	
	# Plot true PDF of the target
	xs = LinRange(-4, 4, 300)
	lines!(ax, xs, pdf.(Cauchy(0, 1), xs); linewidth=2, color=:black, label="True PDF")

	ylims!(0.0, 1.0)
	xlims!(-15, 15)
	
	axislegend(ax)
	fig
end

# ╔═╡ 3c5cd5ff-9efc-4444-9738-15ce5cb82c9a
# ╠═╡ disabled = true
#=╠═╡
#@bind step Clock()
  ╠═╡ =#

# ╔═╡ 70f40d28-c9bf-4bb4-a2bf-64e249dcfbcd
step

# ╔═╡ 74e4eb9c-ad32-4505-8c1d-44212233ec9e
begin
	local xs = LinRange(-4, 4, 400)
	pdf_vals = pdf.(Normal(0, 1), xs)
	
	local fig = Figure(size=(700, 400))
	local ax = Axis(fig[1, 1], xlabel="x", ylabel="Density", title="MCMC Interactive Trace")

	colormap=[ColorSchemes.batlow[i/num_samples] for i=1:num_samples]
	
	lines!(ax, xs, pdf_vals; linewidth=2, color=:black, label="True PDF")
	
	# Trace so far
	lines!(ax, trace[1:step], pdf.(Normal(0,1), trace[1:step]); color=colormap[1:step], linewidth=1.5)
	
	# Current sample
	scatter!(ax, [trace[step]], [pdf(Normal(0,1), trace[step])]; color=:red, markersize=15)

	xlims!(-4, 4)
	
	fig
end

# ╔═╡ Cell order:
# ╠═05e9d630-1f9e-11f0-2374-bdcd5b94dc21
# ╟─44233307-a3df-4bc8-a1cd-e99b61999950
# ╟─24dcb6f1-25cc-4175-8d9f-264cebe86008
# ╠═aa1503f6-ea8d-43f8-96a4-e53ab51ada11
# ╠═517aa4e2-137d-4d2d-8f17-1b98a8aa81e7
# ╠═d8ef1b09-ebfe-4142-b3b4-39b9edfdb38a
# ╠═a30d593e-5ba3-45c6-9ce2-d9f67903a485
# ╠═2d466e44-d3ab-491a-92d6-c11d3cb660ca
# ╠═f2f3b4e5-aac4-491b-a47b-3d1fadb08b33
# ╠═12b931b8-7818-48bb-a1ab-4fbd4e44cb43
# ╠═ab11cf92-a81f-4ee7-9dc4-b13922bec273
# ╠═3c5cd5ff-9efc-4444-9738-15ce5cb82c9a
# ╠═70f40d28-c9bf-4bb4-a2bf-64e249dcfbcd
# ╟─74e4eb9c-ad32-4505-8c1d-44212233ec9e

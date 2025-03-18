### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 285d575a-ad5d-401b-a8b1-c5325e1d27e9
begin
	import Pkg;Pkg.activate()
	
	using CairoMakie, LinearAlgebra, Turing, SpecialFunctions, ColorSchemes, DataFrames, StatsBase, PlutoUI
end

# ╔═╡ 54b50777-cfd7-43a3-bcc2-be47f117e635
TableOfContents()

# ╔═╡ 849ef8ce-4562-4353-8ee5-75d28b1ac929
md"# forward model (also ground-truth)"

# ╔═╡ 0d3b6020-a26d-444e-8601-be511c53c002
md"known parameters."

# ╔═╡ 064eb92e-5ff0-436a-8a2b-4a233ca4fa42
begin
	# size of search space
	L = 50.0 # m
	
	# velocity of wind
	v = [-5.0, 15.0] # m/s
	
	# diffusion coefficient
	D = 25.0 # m²/min
	
	# lifespan
	τ = 50.0 # min

	# κ value, derived from v, D, τ.
	κ = sqrt((dot(v, v) + 4 * D / τ) / (4 * D ^ 2)) # m⁻²
end

# ╔═╡ 7bc01684-4951-4b3e-bb27-96ad99ee0f72
md"unknown params we treat as hidden"

# ╔═╡ 7a9125b3-6138-4d9f-a853-5057075443c1
begin
	# source location
	x₀ = [25.0, 4.0] # m
	
	# source strength
	R = 10.0 # g/min
end

# ╔═╡ b6bfe2c4-e919-4a77-89bc-35d6d9f116ee
md"view c as a function of x₀ and R, the unknowns."

# ╔═╡ a2fb66d1-ae8a-46ca-ab56-5ba934c22360
function c(x::Vector{Float64}, x₀, R) # no type assertions for Turing.jl
	# units R / d [=] g/min / (m²/min) [] = g/m²
	return R / (2 * π * D) * besselk(0, κ * norm(x - x₀)) * 
	        exp(dot(v, x - x₀) / (2 * D))
end

# ╔═╡ 0d35098d-4728-4a03-8951-7549067e0384
c(x₀, x₀, R) # warning: diverges at center.

# ╔═╡ 5ecaca5b-f508-46fd-830e-e9492ca7b4ca
md"ground truth"

# ╔═╡ b217f19a-cc8a-4cb3-aba7-fbb70f5df341
begin
	res = 500
	xs = range(0.0, L, length=res) # m

	cs = [c([x₁, x₂], x₀, R) for x₁ in xs, x₂ in xs] # g/m²
end

# ╔═╡ 0fa42c7c-3dc5-478e-a1d5-8926b927e254
begin
	colormap = ColorScheme(
	    vcat(
	        ColorSchemes.grays[end],
	        reverse([ColorSchemes.viridis[i] for i in 0.0:0.05:1.0])
	    )
	)
	    
	fig = Figure()
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x₁", 
	    ylabel="x₂"
	)
	hm  = heatmap!(xs, xs, cs, colormap=colormap, colorrange=(0.0, maximum(cs)))
	Colorbar(fig[1, 2], hm, label = "concentration c(x₁, x₂) [g/m²]")
	fig
end

# ╔═╡ adbb9f2d-f4f9-4a20-ab52-ccc03358e058
md"
# simulate measurements
measurement noise"

# ╔═╡ c5bcd369-b04c-47d7-b85e-3d51b04b7506
σ = 0.005 # g/m², the measurement noise

# ╔═╡ 95e834f4-4530-4a76-b8a8-f39bb7c0fdb1
function measure_concentration(x::Vector{Float64})
	return c(x, x₀, R) + randn() * σ
end

# ╔═╡ 2818e9b8-8328-4759-ba7a-638ed28329d0
measure_concentration([20.0, 10.0])

# ╔═╡ b9aec8d8-688b-42bb-b3a4-7d04ee39e2ad
md"# simulate robot taking a path and measuring concentration"

# ╔═╡ 50e623c0-49f6-4bb5-9b15-c0632c3a88fd
begin
	Δx = 2.0 # m (step length for robot)
	robot_path = [[0.0, 0.0]] # begin at origin

	function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol)
		if direction == :left
			Δ = [-Δx, 0.0]
		elseif direction == :right
			Δ = [Δx, 0.0]
		elseif direction == :up
			Δ = [0.0, Δx]
		elseif direction == :down
			Δ = [0.0, -Δx]
		else
			error("direction not valid")
		end

		push!(robot_path, robot_path[end] + Δ)
	end

	function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol, n::Int)
		for i = 1:n
			move!(robot_path, direction)
		end
	end

	move!(robot_path, :up, 5)
	move!(robot_path, :right, 7)
	# move!(robot_path, :up, 3)
	
	data = DataFrame(
		"time" => 1:length(robot_path),
		"x [m]" => robot_path,
		"c [g/m²]" => [measure_concentration(x) for x in robot_path]
	)
end

# ╔═╡ deae0547-2d42-4fbc-b3a9-2757fcfecbaa
function viz_data(data::DataFrame)	    
	fig = Figure()
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x₁", 
	    ylabel="x₂"
	)
	sc = scatter!(
		[row["x [m]"][1] for row in eachrow(data)],
		[row["x [m]"][2] for row in eachrow(data)],
		color=[row["c [g/m²]"][1] for row in eachrow(data)],
		colormap=colormap,
		strokewidth=2
	)
	xlims!(0-Δx, L+Δx)
	ylims!(0-Δx, L+Δx)
	Colorbar(fig[1, 2], sc, label="concentration c(x₁, x₂) [g/m²]")
	fig
end

# ╔═╡ d38aeeca-4e5a-40e1-9171-0a187e84eb69
viz_data(data)

# ╔═╡ 26a4354f-826e-43bb-9f52-eea54cc7e30f
R_max = 100.0 # g/min

# ╔═╡ 1e7e4bad-16a0-40ee-b751-b2f3664f6620
@model function plume_model(data)
    #=
	prior distributions
	=#
	# source location
    x₀ ~ filldist(Uniform(0.0, L), 2)
	# source strength
	R ~ Uniform(0.0, R_max)

    #=
	likelihood
		(loop thru observations)
	=#
    for i in 1:nrow(data)
        data[i, "c [g/m²]"] ~ Normal(
			c(data[i, "x [m]"], x₀, R), σ
		)
    end

    return nothing
end

# ╔═╡ c8f33986-82ee-4d65-ba62-c8e3cf0dc8e9
md"# posterior

infer the source location and strength.
"

# ╔═╡ e63481a3-a50a-45ae-bb41-9d86c0a2edd0
begin
	prob_model = plume_model(data)
			
	nb_samples = 1000 # per chain
	nb_chains = 4      # independent chains
	chain = DataFrame(
		sample(prob_model, NUTS(), MCMCSerial(), nb_samples, nb_chains)
	)
end

# ╔═╡ 2fe974fb-9e0b-4c5c-9a5a-a5c0ce0af065
scatter(
	chain[:, "x₀[1]"], chain[:, "x₀[2]"], marker=:+
)

# ╔═╡ 10fe24bf-0c21-47cc-85c0-7c3d7d77b78b
md"### create empirical dist'n for source location"

# ╔═╡ 065befd1-f652-4925-b1b2-4e847a3884dd
# from edges compute centers of bins.
function edges_to_centers(edges)
	n = length(edges)
	return [(edges[i] + edges[i+1]) / 2 for i = 1:n-1]
end

# ╔═╡ 70c50c11-9e20-409c-8c61-ffd744bf45a7
x_edges = collect(0.0:Δx:L+2) .- Δx/2

# ╔═╡ e7567ef6-edaa-4061-9457-b04895a2fca2
x_bin_centers = edges_to_centers(x_edges)

# ╔═╡ f3210949-996f-40b3-afb4-72b665084fa0
hist_x₀ = fit(
	Histogram, (chain[:, "x₀[1]"], chain[:, "x₀[2]"]), (x_edges, x_edges)
)

# ╔═╡ 29b702de-747a-4af7-8474-c1ef0df2c2a3
P = hist_x₀.weights / sum(hist_x₀.weights) # posterior distribution

# ╔═╡ bd0a5555-cbe5-42ae-b527-f62cd9eff22f
heatmap(x_bin_centers, x_bin_centers, P)

# ╔═╡ 71218d02-8e94-45ef-8493-73b7ef437340
function viz_posterior(hist_R, hist_x₀)
	fig = Figure()

	# dist'n of R
	ax_t = Axis(fig[1, 1], xlabel="R [g/L]", ylabel="density")
	R_bin_centers = edges_to_centers(hist_R.edges[1])
	stem!(R_bin_centers, hist_R.weights)

	# dist'n of x₀
	# ax_b = Axis(
	# 	fig[2, 1], xlabel="x₁ [m]", ylabel="x₂ [m]", aspect=DataAspect()
	# )
	# xlims!(ax_b, 0, L)
	# ylims!(ax_b, 0, L)
	# hb = hexbin!(
	# 	ax_b, chain[:, "x₀[1]"], chain[:, "x₀[2]"], colormap=colormap, bins=round(Int, L/Δx)
	# )
	# Colorbar(fig[2, 2], hb, label="density")

	# # show ground-truth
	# vlines!(ax_t, R, color="red", linestyle=:dash)
	# scatter!(ax_b, [x₀[1]], [x₀[2]], marker=:+, color="red")
	fig
end

# ╔═╡ f4d234f9-70af-4a89-9a57-cbc524ec52b4
function viz_posterior(chain::DataFrame)
	fig = Figure()

	# dist'n of R
	ax_t = Axis(fig[1, 1], xlabel="R [g/L]", ylabel="density")
	hist!(chain[:, "R"])

	# dist'n of x₀
	ax_b = Axis(
		fig[2, 1], xlabel="x₁ [m]", ylabel="x₂ [m]", aspect=DataAspect()
	)
	xlims!(ax_b, 0, L)
	ylims!(ax_b, 0, L)
	hb = hexbin!(
		ax_b, chain[:, "x₀[1]"], chain[:, "x₀[2]"], colormap=colormap, bins=round(Int, L/Δx)
	)
	Colorbar(fig[2, 2], hb, label="count")

	# show ground-truth
	vlines!(ax_t, R, color="red", linestyle=:dash)
	scatter!(ax_b, [x₀[1]], [x₀[2]], marker=:+, color="red")
	fig
end

# ╔═╡ 3438fcc0-adde-4aa8-9c71-f16d3c563e49
viz_posterior(hist_R, hist_x₀)

# ╔═╡ 4bb02313-f48b-463e-a5b6-5b40fba57e81
viz_posterior(chain)

# ╔═╡ e98ea44e-2da3-48e9-be38-a43c6983ed08
md"# infotaxis"

# ╔═╡ 14b34270-b47f-4f22-9ba4-db294f2c029c
md"## entropy calcs"

# ╔═╡ baa90d24-6ab4-4ae8-9565-c2302428e9e7
"""
Returns the total entropy of the probability field matrix.

* `pr_field::Matrix{Float64}`
"""
function entropy(pr_field::Matrix{Float64})
	return sum([-pr_field[i]*log(pr_field[i]) for i in eachindex(pr_field) if pr_field[i]>0.0])
end

# ╔═╡ 5695ee1e-a532-4a89-bad1-20e859016174
"""
Sets probability of finding the source at the robot's current coordinates to 0 and renormalizes the probability field.

* `x::Vector{Float64}` - robot's coordinates
* `pr_field::Matrix{Float64}` - probability field to be updated
"""
function miss_source(x::Vector{Float64}, pr_field::Matrix{Float64})
	pr_field[(Int.(x)...)] = 0.0
	pr_field .= pr_field/sum(pr_field)
	return pr_field
end

# ╔═╡ 5509a7c1-1c91-4dfb-96fc-d5c33a224e73
"""
Calculates H(s|a), the expected entropy upon taking action a in belief state s.

* `pr_field::Matrix{Float64}`
"""
function expected_entropy(robot_path::Vector{Vector{Float64}}, direction::Symbol, pr_field::Matrix{Float64})
	@assert a in (:left, :up, :down, :right)

	# make a copy of robot path, and move it
	test_robot = deepcopy(robot_path)
	move!(test_robot, direction)

	# calculate probability of missing
	prob_miss = 1.0 - pr_field[(Int.(test_robot[end])...)]

	# update probability field as if source wasn't found
	test_map = miss_source(test_robot[end], pr_field)

	#=
	I think this is where we need to calculate the posterior using the test_map as the prior.

	=#

	# expected_entropy = prob_miss * test_map_posterior
	
	return expected_entropy
	
end

# ╔═╡ f04d1521-7fb4-4e48-b066-1f56805d18de
md"## simulate"

# ╔═╡ e278ec3e-c524-48c7-aa27-dd372daea005
"""
TODO:
input should be starting location and a prior. It should check the information gain (entropy reduction) from each possible action and choose the action that reduces entropy the most.
"""
function sim()

	return argmin(expected_entropy())
end

# ╔═╡ Cell order:
# ╠═285d575a-ad5d-401b-a8b1-c5325e1d27e9
# ╠═54b50777-cfd7-43a3-bcc2-be47f117e635
# ╟─849ef8ce-4562-4353-8ee5-75d28b1ac929
# ╟─0d3b6020-a26d-444e-8601-be511c53c002
# ╠═064eb92e-5ff0-436a-8a2b-4a233ca4fa42
# ╟─7bc01684-4951-4b3e-bb27-96ad99ee0f72
# ╠═7a9125b3-6138-4d9f-a853-5057075443c1
# ╟─b6bfe2c4-e919-4a77-89bc-35d6d9f116ee
# ╠═a2fb66d1-ae8a-46ca-ab56-5ba934c22360
# ╠═0d35098d-4728-4a03-8951-7549067e0384
# ╟─5ecaca5b-f508-46fd-830e-e9492ca7b4ca
# ╠═b217f19a-cc8a-4cb3-aba7-fbb70f5df341
# ╠═0fa42c7c-3dc5-478e-a1d5-8926b927e254
# ╟─adbb9f2d-f4f9-4a20-ab52-ccc03358e058
# ╠═c5bcd369-b04c-47d7-b85e-3d51b04b7506
# ╠═95e834f4-4530-4a76-b8a8-f39bb7c0fdb1
# ╠═2818e9b8-8328-4759-ba7a-638ed28329d0
# ╟─b9aec8d8-688b-42bb-b3a4-7d04ee39e2ad
# ╠═50e623c0-49f6-4bb5-9b15-c0632c3a88fd
# ╠═deae0547-2d42-4fbc-b3a9-2757fcfecbaa
# ╠═d38aeeca-4e5a-40e1-9171-0a187e84eb69
# ╠═26a4354f-826e-43bb-9f52-eea54cc7e30f
# ╠═1e7e4bad-16a0-40ee-b751-b2f3664f6620
# ╟─c8f33986-82ee-4d65-ba62-c8e3cf0dc8e9
# ╠═e63481a3-a50a-45ae-bb41-9d86c0a2edd0
# ╠═2fe974fb-9e0b-4c5c-9a5a-a5c0ce0af065
# ╟─10fe24bf-0c21-47cc-85c0-7c3d7d77b78b
# ╠═065befd1-f652-4925-b1b2-4e847a3884dd
# ╠═70c50c11-9e20-409c-8c61-ffd744bf45a7
# ╠═e7567ef6-edaa-4061-9457-b04895a2fca2
# ╠═f3210949-996f-40b3-afb4-72b665084fa0
# ╠═29b702de-747a-4af7-8474-c1ef0df2c2a3
# ╠═bd0a5555-cbe5-42ae-b527-f62cd9eff22f
# ╠═71218d02-8e94-45ef-8493-73b7ef437340
# ╠═3438fcc0-adde-4aa8-9c71-f16d3c563e49
# ╠═f4d234f9-70af-4a89-9a57-cbc524ec52b4
# ╠═4bb02313-f48b-463e-a5b6-5b40fba57e81
# ╟─e98ea44e-2da3-48e9-be38-a43c6983ed08
# ╟─14b34270-b47f-4f22-9ba4-db294f2c029c
# ╠═baa90d24-6ab4-4ae8-9565-c2302428e9e7
# ╠═5695ee1e-a532-4a89-bad1-20e859016174
# ╠═5509a7c1-1c91-4dfb-96fc-d5c33a224e73
# ╟─f04d1521-7fb4-4e48-b066-1f56805d18de
# ╠═e278ec3e-c524-48c7-aa27-dd372daea005

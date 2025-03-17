### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 285d575a-ad5d-401b-a8b1-c5325e1d27e9
begin
	import Pkg;Pkg.activate()
	
	using CairoMakie, LinearAlgebra, Turing, SpecialFunctions, ColorSchemes, DataFrames
end

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
	move!(robot_path, :up, 3)
	
	data = DataFrame(
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

# ╔═╡ 1e7e4bad-16a0-40ee-b751-b2f3664f6620
@model function plume_model(data)
    #=
	prior distributions
	=#
	# source location
    x₀ ~ filldist(Uniform(0.0, L), 2)
	# source strength
	R ~ Uniform(0.0, 100.0)

    #=
	likelihood
		(loop thru observations)
	=#
    for i in 1:nrow(data)
		ĉᵢ = AutoForwardDiff
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
			
	nb_samples = 2_500 # per chain
	nb_chains = 4      # independent chains
	chain = DataFrame(
		sample(prob_model, NUTS(), MCMCSerial(), nb_samples, nb_chains)
	)
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
	Colorbar(fig[2, 2], hb, label="density")

	# show ground-truth
	vlines!(ax_t, R, color="red", linestyle=:dash)
	scatter!(ax_b, [x₀[1]], [x₀[2]], marker=:+, color="red")
	fig
end

# ╔═╡ 4bb02313-f48b-463e-a5b6-5b40fba57e81
viz_posterior(chain)

# ╔═╡ Cell order:
# ╠═285d575a-ad5d-401b-a8b1-c5325e1d27e9
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
# ╟─1e7e4bad-16a0-40ee-b751-b2f3664f6620
# ╟─c8f33986-82ee-4d65-ba62-c8e3cf0dc8e9
# ╠═e63481a3-a50a-45ae-bb41-9d86c0a2edd0
# ╠═f4d234f9-70af-4a89-9a57-cbc524ec52b4
# ╠═4bb02313-f48b-463e-a5b6-5b40fba57e81

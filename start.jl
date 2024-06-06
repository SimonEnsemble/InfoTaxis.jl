### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 0d48c2ef-89e3-4c58-bc22-4eb0d0c42ca3
begin
	import Pkg; Pkg.activate()
	using LinearAlgebra, CairoMakie, Distributions, SpecialFunctions
end

# ╔═╡ 4c942495-c9fa-4673-9cc4-8973dd0a3977
md"# search grid"

# ╔═╡ 18e9885b-3c89-4f1a-8d90-0a99eb5a3bec
begin
	struct SearchGrid
		N::Int             # number of voxels
		L::Float64         # L x L
		Δ::Float64
	end

	SearchGrid(N::Int, L::Float64) = SearchGrid(N, L, L / N)
end

# ╔═╡ 28521bff-b41a-4d27-81f2-3d34ff2e2973
sg = SearchGrid(3, 1.0)

# ╔═╡ a0803a48-38d1-4a4d-9084-612687c89f07
function loc(i::Int, j::Int, sg::SearchGrid) 
	if i > sg.N || j > sg.N
		error()
	end
	return [sg.Δ * (i - 1), sg.Δ * (j - 1)] .+ sg.Δ / 2
end

# ╔═╡ c6657c9a-2170-4455-be65-e0e7c63cabe0
function viz(sg::SearchGrid)
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₁", aspect=DataAspect())
	for i = 1:sg.N+1
		for j = 1:sg.N+1
			lines!([0, sg.L], [(j - 1) * sg.Δ], color="black")
			lines!([(i - 1) * sg.Δ], [0, sg.L], color="black")
			
			if i <= sg.N && j <= sg.N
				x = loc(i, j, sg) 
				scatter!([x[1]], [x[2]], color="black")
			end
		end
	end
	fig
end

# ╔═╡ ae89b913-be1e-4abb-9822-327cebd3b4bc
viz(sg)

# ╔═╡ 89821ed9-82ca-42c3-bd45-dc25596627bd
md"# forward model"

# ╔═╡ e659213c-2452-11ef-0d91-89be83a7d4bd
begin
	struct PlumeParams
	    R::Float64          # source strength [g/min]
	    v::Vector{Float64}  # wind vector [m/s]
	    D::Float64          # diffusion coefficent [m²/min]
	    τ::Float64          # lifespan [min]
	    κ::Float64          # m⁻²
	end
	
	# constructor that computes κ for us
	function PlumeParams(; R=R, D=D, τ=τ, v=v)
	    κ = sqrt((dot(v, v) + 4 * D / τ) / (4 * D ^ 2)) # m⁻²
	    return PlumeParams(R, v, D, τ, κ)
	end
end

# ╔═╡ 5f6089d6-f89b-4d90-b7cc-dda89bf1fd48
function c(x::Vector{Float64}, x₀::Vector{Float64}, p::PlumeParams) # g/m²
	return p.R / (2 * π * p.D) * besselk(0, p.κ * norm(x - p.x₀)) * 
	        exp(dot(p.v, x - p.x₀) / (2 * p.D))
end

# ╔═╡ 5c8bb8b0-ab55-4011-8fe6-70ddea5c08a7
p = PlumeParams(R=10.0, D=25.0, τ=50.0, v=[-5.0, 15.0])

# ╔═╡ 52a1a366-e911-4db8-b09d-3811085a1373
md"# measurement model"

# ╔═╡ cf304c4a-b015-42dc-93d6-a3ea89a049a7
mutable struct Robot
	x::Vector{Float64}
	a::Int      # area of sensor
	Δt::Float64 # measurement time
end

# ╔═╡ 8a415e23-5ebc-49bb-964c-dd6483525b57
function μ(robot::Robot, x₀::Vector{Float64}, p::PlumeParams)
	return p.R * robot.Δt / log(1 / (p.κ * robot.a)) * besselk(0, p.κ * norm(robot.x - p.x₀))
end

# ╔═╡ Cell order:
# ╠═0d48c2ef-89e3-4c58-bc22-4eb0d0c42ca3
# ╟─4c942495-c9fa-4673-9cc4-8973dd0a3977
# ╠═18e9885b-3c89-4f1a-8d90-0a99eb5a3bec
# ╠═28521bff-b41a-4d27-81f2-3d34ff2e2973
# ╠═a0803a48-38d1-4a4d-9084-612687c89f07
# ╠═c6657c9a-2170-4455-be65-e0e7c63cabe0
# ╠═ae89b913-be1e-4abb-9822-327cebd3b4bc
# ╟─89821ed9-82ca-42c3-bd45-dc25596627bd
# ╠═e659213c-2452-11ef-0d91-89be83a7d4bd
# ╠═5f6089d6-f89b-4d90-b7cc-dda89bf1fd48
# ╠═5c8bb8b0-ab55-4011-8fe6-70ddea5c08a7
# ╟─52a1a366-e911-4db8-b09d-3811085a1373
# ╠═cf304c4a-b015-42dc-93d6-a3ea89a049a7
# ╠═8a415e23-5ebc-49bb-964c-dd6483525b57

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

# ╔═╡ 285d575a-ad5d-401b-a8b1-c5325e1d27e9
begin
	import Pkg; Pkg.activate()
	
	using CairoMakie, LinearAlgebra, Turing, SpecialFunctions, ColorSchemes, DataFrames, StatsBase, PlutoUI, Test, Distributions, Printf, PlutoTeachingTools
end

# ╔═╡ 54b50777-cfd7-43a3-bcc2-be47f117e635
TableOfContents()

# ╔═╡ 52d76437-0d60-4589-996f-461eecf0d45d
md"""
# Declarations
"""

# ╔═╡ e5b1369e-0b4b-4da3-9c95-07ceda11b31d
md"## constants"

# ╔═╡ 064eb92e-5ff0-436a-8a2b-4a233ca4fa42
begin
	# size of search space
	L = 1000.0 # m
	Δx = 10.0 # m
	
	# constant attenuation for air
	Σ_air = 0.015
	
	# Detector Parameters
	ϵ = 0.95 #efficiency
	Δt = 1.0 #s
	A = 0.0224 #m^2

	# source parameters
	x₀ = [250.0, 250.0]
	P_γ = 0.85 #about 85% decays emit detectable gamma
	Σ = 0.2 #macroscopic cross section (mean free path)
	mCi = 0.050 #50 μCi
	I = mCi * 3.7 * 10^7 * P_γ # 1mCi = 3.7*10^7 Bq
	#counts/gamma - multiply this by the value normalized to #of photons

	# colors
	colormap = reverse(vcat([ColorSchemes.hot[i] for i in 0.0:0.02:1], ColorSchemes.batlow[0.0]))

	# Locate files for which data needs to be extracted
	dir_name = "sim_data"
	data_dir = joinpath(pwd(), dir_name)
	data_files = [joinpath(data_dir, file) for file in readdir(data_dir)]

	# Turing parameters
	I_max = 1e11 #emmissions/s

end

# ╔═╡ b8d6c195-d639-4438-8cab-4dcd99ea2547
function attenuation_constant(x::Vector{Float64}, x₀; Σ::Float64=Σ_air)
    distance = norm(x .- x₀)
    return exp(-Σ * distance)
end

# ╔═╡ 82577bab-5ce9-4164-84db-9cfa28b501b0
md"""
## Structs
"""

# ╔═╡ a9c65b66-eb92-4943-91a9-9d0ea6cfa3d3
md"### obstructions"

# ╔═╡ 57478c44-578e-4b53-b656-c9b3c208dec4
begin
	abstract type Obstruction end
	
	struct Rectangle <: Obstruction
	    center::Tuple{Float64, Float64}
	    width::Float64
	    height::Float64
	end
	
	struct Circle <: Obstruction
	    center::Tuple{Float64, Float64}
	    radius::Float64
	end
	
	struct Polygon <: Obstruction
	    vertices::Vector{Tuple{Float64, Float64}}
	end
end

# ╔═╡ 03910612-d3fe-481c-bb70-dd5578bd8258
md"### rad sim"

# ╔═╡ dd357479-64ef-4823-8aba-931323e89aed
struct RadSim
    γ_matrix::Vector{Matrix{Float64}}
    Δxy::Float64
    Δz::Float64
    Lxy::Float64
    Lz::Float64
end

# ╔═╡ 7fcecc0e-f97c-47f7-98db-0da6d6c1811e
md"""
# Data Import
"""

# ╔═╡ 19c95a83-670c-4ad6-82a1-5a4b6809f1d4
function extract_parameters(data::Dict)

	parameters = Dict()

	#Δx,y and L parameters
	Δxy = data["y_bin_bounds"][2] - data["y_bin_bounds"][1]
	parameters["Δxy"] = Δxy
	Δz = data["z_bin_bounds"][2] - data["z_bin_bounds"][1]
	parameters["Δz"] = Δz 
	parameters["Lxy"] = (length(data["x_bin_bounds"])-1) * Δxy
	parameters["Lz"] = (length(data["z_bin_bounds"])-1) * Δz

	#assert square grid
	@assert Δxy == data["x_bin_bounds"][2] - data["x_bin_bounds"][1] "x and y should have the same grid spacing!"
	

	norm_gamma_matrix = zeros(length(data["x_bin_bounds"])-1,
							 length(data["y_bin_bounds"])-1,
							 length(data["z_bin_bounds"])-1
							 )


	for row in eachrow(data["energy_field_data"])
		x_start = data["x_bin_bounds"][1]
		y_start = data["y_bin_bounds"][1]
		z_start = data["z_bin_bounds"][1]
	    # Compute indices from coordinates
	    i = Int(round((row["X"] - x_start) / Δxy + 0.5))
	    j = Int(round((row["Y"] - y_start)  / Δxy + 0.5))
	    k = Int(round((row["Z"] - z_start)  / Δz + 0.5))
	
		norm_gamma_matrix[i, j, k] = row["Result"]
	end

	parameters["γ_matrix"] = norm_gamma_matrix

	return parameters
end

# ╔═╡ e62ba8da-663a-4b58-afe6-910710d7518e
function import_data(data_file::String)
	#data_of_interest = "collecting boxes"
	@assert isfile(data_file) "$(data_file) is not a file!!!"
	
	flags = Dict(
		"tally_bin_bounds" => false,
		"x_dir" => false,
		"y_dir" => false,
		"z_dir" => false,
		"energy_grid" => false
	)

	data = Dict(
		"x_bin_bounds" => Array{Float64}(undef, 0),
		"y_bin_bounds" => Array{Float64}(undef, 0),
		"z_bin_bounds" => Array{Float64}(undef, 0),
		"energy_field_data" => DataFrame(
			Energy = Array{Float64}(undef, 0),
			X = Array{Float64}(undef, 0),
			Y = Array{Float64}(undef, 0),
			Z = Array{Float64}(undef, 0),
			Result = Array{Float64}(undef, 0),
			RelError = Array{Float64}(undef, 0)
		)
	)

	
	open(data_file) do f
		while !eof(f)
			#read the line
			f_line = lowercase(readline(f))
			# replace double+ spaces with single spaces
			s = replace(f_line, r"\s{2,}" => " ")

			#check for first line of data
			if contains(s, "tally bin boundaries")
				flags["tally_bin_bounds"] = true
				continue
			elseif contains(s, "energy x y")
				flags["energy_grid"] = true
				continue
			end

			if flags["tally_bin_bounds"]
				if contains(s, "energy bin boundaries")
					flags["tally_bin_bounds"] = false
					continue
				end
   				for dir in ["x", "y", "z"]
        			if occursin("$dir direction", s)
            			flags["x_dir"] = flags["y_dir"] = flags["z_dir"] = false
           				flags["$(dir)_dir"] = true
						
						values = parse.(Float64, split(s)[3:end])
						data["$(dir)_bin_bounds"] = values
					elseif !(occursin("direction", s))
						if flags["$(dir)_dir"]
							values = parse.(Float64, split(s))
							data["$(dir)_bin_bounds"] = vcat(data["$(dir)_bin_bounds"], values)
						end
					end
				end
			end

			if flags["energy_grid"]
				values = parse.(Float64, split(s))
				push!(data["energy_field_data"], values)
			end
		end
	end

	parameters = extract_parameters(data)

	rad_sim = RadSim(
		[parameters["γ_matrix"][:, :, i] for i in 1:size(parameters["γ_matrix"], 3)],
		parameters["Δxy"],
		parameters["Δz"],
		parameters["Lxy"],
		parameters["Lz"]
	)
	return rad_sim
end

# ╔═╡ 1197e64f-34c2-4892-8da5-3b26ee6e7c2f
begin
	num_sims = length(data_files)
	rad_sim_data = [import_data(data_files[i]) for i=1:num_sims]

	test_rad_sim = rad_sim_data[1]
end

# ╔═╡ 7278adb5-2da1-4ea1-aa38-d82c23510242
md"""
## `Visualize` - Imported Data
"""

# ╔═╡ 7211ea6e-6535-4e22-a2ef-a1994e81d22a
function viz_model_data!(ax, rad_sim::RadSim; z_slice::Int64=1)
	
	scale = ReversibleScale(
	    x -> log10(x + 1),   # forward: avoids log(0)
	    x -> 10^x - 1        # inverse
	)

	#build colormap with black at around 0 counts
	#colormap = reverse(vcat([ColorSchemes.hot[i] for i in 0.0:0.02:1], ColorSchemes.batlow[0.0]))

	# x and y values 
	xs = ys = [val for val in 0:rad_sim.Δxy:rad_sim.Lxy]
	#counts
	counts_I = I * rad_sim.γ_matrix[z_slice]

	hm = heatmap!(ax, xs, ys, counts_I, colormap=colormap, colorscale = scale)
	return hm
end

# ╔═╡ 63c8b6dd-d12a-42ec-ab98-1a7c6a991dbd
function viz_model_data(rad_sim::RadSim; z_slice::Int64=1)
	fig = Figure()
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x₁", 
	    ylabel="x₂"
	)

	counts_I = I * rad_sim.γ_matrix[z_slice]
	
	hm = viz_model_data!(ax, rad_sim)

	#establish logarithmic colorbar tick values
	colorbar_tick_values = [10.0^e for e in range(0, log10(maximum(counts_I)), length=6)]
	colorbar_tick_values[1] = 0.0
	colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]

	Colorbar(fig[1, 2], hm, label = "counts / s", ticks = (colorbar_tick_values, colorbar_tick_labels))
	
	fig
end

# ╔═╡ 1f12fcd1-f962-45e9-8c07-4f42f869d6a0
viz_model_data(test_rad_sim)

# ╔═╡ 7ed3ac02-2915-4995-8ff4-ea781b283f7f
argmax(test_rad_sim.γ_matrix[1])

# ╔═╡ 849ef8ce-4562-4353-8ee5-75d28b1ac929
md"# Analytical (Poisson) Model"

# ╔═╡ e622cacd-c63f-416a-a4ab-71ba9d593cc8
"""
Generates a Poisson distribution based on position, source position and source strength.

* `x::Vector{Float64}` - Measurement/true value location.
* `x₀::Vector{Float64}` - source location.
* `I::Float64` - source strength in Bq.
"""
function count_Poisson(x::Vector{Float64}, x₀, I)
	distance = norm(x₀ .- x)
	attenuation = attenuation_constant(x, x₀)
	λ = I * Δt * ϵ * (A / (4π * distance^2)) * exp(-attenuation)
	
	if isnan(λ) || λ < 0
		return Poisson(0.0)
	else
		return Poisson(λ)
	end
end

# ╔═╡ 8ed5d321-3992-40db-8a2e-85abc3aaeea0
md"""
## `Visualize` - Analytical Model
"""

# ╔═╡ 0175ede7-b2ab-4ffd-8da1-278120591027
function viz_c_analytical!(ax, color_scale; res::Int=500, L::Float64=1000.0, x₀::Vector{Float64}=[251.0, 251.0], I::Float64=1.16365e10, source::Union{Nothing, Vector{Float64}}=nothing, scale_max::Float64=1e6)
	#colormap = reverse([ColorSchemes.hot[i] for i in 0.0:0.05:1])

	xs = collect(0.0:Δx:L)
	counts = [mean(count_Poisson([x₁, x₂], x₀, I)) for x₁ in xs, x₂ in xs] # counts

	hm = heatmap!(ax, xs, xs, counts, colormap=colormap, colorscale = color_scale, colorrange=(0, scale_max))

	if ! isnothing(source)
		scatter!(ax, [source[1]], [source[2]], color="red", marker=:xcross, markersize=15, label="source", strokewidth=1)
	end

	return hm, counts
end

# ╔═╡ 6fa37ac5-fbc2-43c0-9d03-2d194e136951
function viz_c_analytical(; res::Int=500, L::Float64=1000.0, x₀::Vector{Float64}=[251.0, 251.0], I::Float64=1.16365e10, source::Union{Nothing, Vector{Float64}}=nothing, scale_max::Float64=1e5)
	fig = Figure()
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x", 
	    ylabel="y"
	)

	scale = ReversibleScale(
	    x -> log10(x + 1),   # forward: avoids log(0)
	    x -> 10^x - 1        # inverse
	)

	hm, _ = viz_c_analytical!(ax, scale, res=res, L=L, x₀=x₀, I=I, source=source, scale_max=scale_max)

	colorbar_tick_values = [10.0^e for e in range(0, log10(scale_max), length=6)]
	colorbar_tick_values[1] = 0.0
	colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]
	
	#tick_pos = scale_option_2.(colorbar_tick_values)

	Colorbar(fig[1, 2], hm, label = "counts / s", ticks = (colorbar_tick_values, colorbar_tick_labels))
	
	fig
end

# ╔═╡ bc761684-699f-430b-bcb9-9b03410fa1d5
mean(Poisson(5.0))

# ╔═╡ 5b9aaaeb-dbb3-4392-a3a1-ccee94d75fed
viz_c_analytical(scale_max = maximum(I * test_rad_sim.γ_matrix[1]))

# ╔═╡ 31864185-6eeb-4260-aa77-c3e94e467558
md"# Simulate Movement"

# ╔═╡ 015b9f4d-09b8-49f3-bc03-2fd3b972e933
md"## sample environment"

# ╔═╡ 126df6ec-9074-4712-b038-9371ebdbc51d
function sample_model(x::Vector{Float64}, rad_sim::RadSim; I::Float64=I, Δx::Float64=Δx, z_index::Int=1)
	counts_I = I * rad_sim.γ_matrix[z_index]
	@assert count([round(Int, x[i] / Δx) <= size(counts_I, i) && x[i] >= 0.0 for i=1:2]) == 2 "r coordinate values outside of domain"

	#add background noise
	λ_background = 1.5
	noise = rand(Poisson(λ_background)) * rand([-1, 1])
	
	measurement = counts_I[round(Int, x[1] / Δx)+1, round(Int, x[2] / Δx)+1, z_index] + noise
	measurement = max(measurement, 0)

	return round(Int, measurement)
end

# ╔═╡ bfe17543-7b54-4f52-9679-f723adafdbdd
md"## movement"

# ╔═╡ c6c39d74-4620-43df-8eb1-83c436924530
function get_Δ(direction::Symbol; Δx::Float64=Δx)
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

	return Δ
end

# ╔═╡ 3c0f8b63-6276-488c-a376-d5c554a5555d
	function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol; Δx::Float64=Δx)
		Δ = get_Δ(direction, Δx=Δx)
		push!(robot_path, robot_path[end] + Δ)
	end

# ╔═╡ ddc23919-17a7-4c78-86f0-226e4d447dbe
	function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol, n::Int; Δx::Float64=Δx)
		for i = 1:n
			move!(robot_path, direction, Δx=Δx)
		end
	end

# ╔═╡ 50e623c0-49f6-4bb5-9b15-c0632c3a88fd
begin
	#Δx = 10.0 # m (step length for robot)
	robot_path = [[0.0, 0.0]] # begin at origin
	
	move!(robot_path, :up, 5)
	move!(robot_path, :right, 7)
	# move!(robot_path, :up, 3)
	
	data = DataFrame(
		"time" => 1:length(robot_path),
		"x [m]" => robot_path,
		"counts" => [sample_model(x, test_rad_sim, I=I) for x in robot_path]
	)
end

# ╔═╡ de738002-3e80-4aab-bedb-f08533231ed7
md"## `Visualize` - Movement and Measurement"

# ╔═╡ 82425768-02ba-4fe3-ab89-9ac95a45e55e
function viz_path!(ax, path_data::DataFrame; scale_max::Float64=1e6)

	positions = [(row["x [m]"][1], row["x [m]"][2]) for row in eachrow(data)]
	if length(positions) > 1
	    color_map = reverse([ColorSchemes.hot[i] for i in range(0, 1.0, length=length(positions))])
	    line_colors = [get(reverse(ColorSchemes.winter), i) for i in range(0, stop=1.0, length=length(positions))]
	    line_widths = collect(reverse(range(0.5, stop=6.0, length=length(positions))))
	
	    for i in 1:length(positions)-1
	        r1, y1 = positions[i]
	        r2, y2 = positions[i+1]
	        lines!(ax, [r1, r2], [y1, y2], color=line_colors[i], linewidth=line_widths[i], joinstyle = :round)
	    end
	end

	scale = ReversibleScale(
		    x -> log10(x + 1),   # forward: avoids log(0)
		    x -> 10^x - 1        # inverse
		)

	sc = scatter!(
			[row["x [m]"][1] for row in eachrow(path_data)],
			[row["x [m]"][2] for row in eachrow(path_data)],
			color=[row["counts"][1] for row in eachrow(path_data)],
			colormap=colormap,
			colorscale = scale,
			colorrange=(0.0, scale_max),
			strokewidth=2,
			markersize=11
		)
end

# ╔═╡ deae0547-2d42-4fbc-b3a9-2757fcfecbaa
function viz_data_collection(path_data::DataFrame; x₀::Union{Nothing, Vector{Float64}}=nothing, rad_sim::Union{Nothing, RadSim}=nothing, res::Float64=1000.0, L::Float64=1000.0, scale_max::Float64=1e6, z_slice::Int64=1, save_num::Int64=0)	    
	fig = Figure(size=(res, res))
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x₁", 
	    ylabel="x₂",
		xlabelsize=40,
		ylabelsize=40,
		xticklabelsize=21,
		yticklabelsize=21
	)

	if ! isnothing(rad_sim)
		#get source, grid size and scale_max from data
		counts_I = I * rad_sim.γ_matrix[z_slice]
		source_coord = argmax(counts_I)
		source = [coord * rad_sim.Δxy for coord in source_coord.I]
		source[1] = source[1] - Δx
		scale_max = maximum(counts_I)
		L = rad_sim.Lxy
		
		scale = ReversibleScale(
		    x -> log10(x + 1),   # forward: avoids log(0)
		    x -> 10^x - 1        # inverse
		)
		
		
		hm = viz_model_data!(ax, rad_sim)

		colorbar_tick_values = [10.0^e for e in range(0, log10(scale_max), length=6)]
		colorbar_tick_values[1] = 0.0

		colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]

		Colorbar(fig[1, 2], hm, label = "counts [counts/s]", ticks = (colorbar_tick_values, colorbar_tick_labels), ticklabelsize=25, labelsize=35)
	
	end

	viz_path!(ax, path_data, scale_max=scale_max)

	if ! isnothing(x₀) || ! isnothing(rad_sim)
		scatter!([source[1]], [source[2]], color="black", marker=:xcross, markersize=25, label="source", strokewidth=1)

		#axislegend(location=:tr)
	end
	
	xlims!(0-Δx, L+Δx)
	ylims!(0-Δx, L+Δx)
	
	if isnothing(rad_sim)
		Colorbar(fig[1, 2], sc, label="counts")
	end

	if save_num > 0
		save("$(save_num).png", fig)
	end
	
	fig
end

# ╔═╡ 9fbe820c-7066-40b5-9617-44ae0913928e
viz_data_collection(data, rad_sim=test_rad_sim, res=1000.0)

# ╔═╡ 3ae4c315-a9fa-48bf-9459-4b7131f5e2eb
md"# Turing MCMC"

# ╔═╡ c6783f2e-d826-490f-93f5-3da7e2717a02
md"## naive rad model"

# ╔═╡ 1e7e4bad-16a0-40ee-b751-b2f3664f6620
@model function rad_model(data)
    #=
	prior distributions
	=#
	# source location
    x₀ ~ filldist(Uniform(0.0, L), 2)
	# source strength
	I ~ Uniform(0.0, I_max)

    #=
	likelihood
		(loop thru observations)
	=#
    for i in 1:nrow(data)
        data[i, "counts"] ~ count_Poisson(data[i, "x [m]"], x₀, I)
    end

    return nothing
end

# ╔═╡ e63481a3-a50a-45ae-bb41-9d86c0a2edd0
begin
	#
	prob_model = rad_model(data)
	nb_samples = 4000 # per chain
	nb_chains = 1      # independent chains
	chain = DataFrame(
		sample(prob_model, NUTS(), MCMCSerial(), nb_samples, nb_chains)
	)
end

# ╔═╡ 21486862-b3c2-4fcc-98b2-737dcc5211fb
md"## `Visualize` - Turing chain"

# ╔═╡ 0a39daaa-2c20-471d-bee3-dcc06554cf78
function viz_chain_data!(ax, chain; show_source::Bool=true)

	scatter!(ax,
		chain[:, "x₀[1]"], chain[:, "x₀[2]"], marker=:+
	)
	if show_source
		scatter!(ax, x₀[1], x₀[2], color="red", label="source", marker=:xcross, markersize=15, strokewidth=1)
		axislegend(ax, location=:tr)
	end
end

# ╔═╡ 2fe974fb-9e0b-4c5c-9a5a-a5c0ce0af065
function viz_chain_data(chain; res::Float64=500.0, L::Float64=L, show_source::Bool=true)
	fig = Figure(size = (res, res))
	ax = Axis(fig[1, 1])

	viz_chain_data!(ax, chain, show_source=show_source)

	xlims!(-1, L+1)
	ylims!(-1, L+1)
	
	return fig
end

# ╔═╡ ea2dc60f-0ec1-4371-97f5-bf1e90888bcb
 viz_chain_data(chain)

# ╔═╡ 65d603f4-4ef6-4dff-92c1-d6eef535e67e
md"## `Visualize` - Turing chain heatmap"

# ╔═╡ 6da11e28-c276-4f45-a1aa-2c86ab26c85a
function chain_to_P(chain::DataFrame; Δx::Float64=Δx, L::Float64=L)
	x_edges = collect(0.0:Δx:L+2) .- Δx/2
	hist_x₀ = fit(
		Histogram, (chain[:, "x₀[1]"], chain[:, "x₀[2]"]), (x_edges, x_edges)
	)
	return hist_x₀.weights / sum(hist_x₀.weights)
end

# ╔═╡ 50830491-6285-4915-b59a-fa5bb7298e51
function x_to_bin(x::Float64; Δx::Float64=Δx, L::Float64=L)
	x_edges = collect(0.0:Δx:L+Δx) .- Δx/2
	
	for b = 1:length(x_edges)-1
		if x < x_edges[b+1]
			return b
		end
	end
end

# ╔═╡ 065befd1-f652-4925-b1b2-4e847a3884dd
function edges_to_centers(; Δx::Float64=Δx, L::Float64=L)
	x_edges = collect(0.0:Δx:L+Δx) .- Δx/2
	n = length(x_edges)
	return [(x_edges[i] + x_edges[i+1]) / 2 for i = 1:n-1]
end

# ╔═╡ e1303ce3-a8a3-4ac1-8137-52d32bf222e2
P = chain_to_P(chain)

# ╔═╡ e7567ef6-edaa-4061-9457-b04895a2fca2
x_bin_centers = edges_to_centers()

# ╔═╡ bd0a5555-cbe5-42ae-b527-f62cd9eff22f
heatmap(x_bin_centers, x_bin_centers, P)

# ╔═╡ aa72cf61-839d-4707-95c8-0a9230e77d56
md"## `Visualize` - Posterior"

# ╔═╡ f4d234f9-70af-4a89-9a57-cbc524ec52b4
function viz_posterior(chain::DataFrame)
	fig = Figure()

	# dist'n of I
	ax_t = Axis(fig[1, 1], xlabel="I [g/L]", ylabel="density", xscale=log10)
	hist!(ax_t, chain[:, "I"], bins=[10.0^e for e in range(0, log10(I_max), length=50)])
	#xscale!(ax_t, :log10)

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
	vlines!(ax_t, I, color="red", linestyle=:dash)
	scatter!(ax_b, [x₀[1]], [x₀[2]], marker=:+, color="red")
	fig
end

# ╔═╡ 4bb02313-f48b-463e-a5b6-5b40fba57e81
viz_posterior(chain)

# ╔═╡ 325d565d-ef0e-434a-826a-adb68825f0fd
TODO("EVERYTHING BELOW HERE TO BE REWORKED")

# ╔═╡ c8f33986-82ee-4d65-ba62-c8e3cf0dc8e9
md"# posterior

infer the source location and strength.
"

# ╔═╡ c37e3d82-2320-4278-8b9b-24912a93fd96
md"""
## Viz Chain
"""

# ╔═╡ 10fe24bf-0c21-47cc-85c0-7c3d7d77b78b
md"### create empirical dist'n for source location"

# ╔═╡ 06ecc454-9cd5-432d-bc1c-b269ee3f0794
chain

# ╔═╡ e98ea44e-2da3-48e9-be38-a43c6983ed08
md"# infotaxis & Thompson sampling"

# ╔═╡ 14b34270-b47f-4f22-9ba4-db294f2c029c
md"## entropy calcs"

# ╔═╡ baa90d24-6ab4-4ae8-9565-c2302428e9e7
"""
entropy of belief state H(s)
(i.e. entropy of posterior over source location)
"""
entropy(P::Matrix{Float64}) = sum(
	[-P[i] * log2(P[i]) for i in eachindex(P) if P[i] > 0.0]
)

# ╔═╡ f04d1521-7fb4-4e48-b066-1f56805d18de
md"## simulate"

# ╔═╡ 83052e75-db08-4e0a-8c77-35487c612dae
function pos_to_index(pos::Vector{Float64}; Δx::Float64=2.0)
    x₁ = Int(floor((pos[1] + 1) / Δx)) + 1
    x₂ = Int(floor((pos[2] + 1) / Δx)) + 1
    return (x₁, x₂)
end

# ╔═╡ 5695ee1e-a532-4a89-bad1-20e859016174
"""
Sets probability of finding the source at the robot's current coordinates to 0 and renormalizes the probability field.

* `r::Vector{Float64}` - robot's coordinates
* `pr_field::Matrix{Float64}` - probability field to be updated
"""
function miss_source(r::Vector{Float64}, pr_field::Matrix{Float64})
	indicies = pos_to_index(r)
	pr_field[(indicies...)] = 0.0
	pr_field .= pr_field/sum(pr_field)
	return pr_field
end

# ╔═╡ 5509a7c1-1c91-4dfb-96fc-d5c33a224e73
"""
H(s|a), the expected entropy of r₀ *after* taking action a in belief state s.

a ∈ {left, right, up, down}

i.e. the expected entropy of successor belief states s':
 Σ s′ P(s′ | s , a) H(s′)

essentially, we *simulate* taking action a, where the robot takes a move and takes a measurement, then compute the entropy of the posterior after that measurement.

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `direction::Symbol` - for entropy reduction calculation, which direction should the robot consider going.
* `pr_field::Matrix{Float64}` - current posterior for the source location.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
* `data::DataFrame` - current data frame containing sample data.
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
"""
function expected_entropy(
	robot_path::Vector{Vector{Float64}}, 
	direction::Symbol, 
	pr_field::Matrix{Float64},
	chain::DataFrame,
	data::DataFrame;
	num_mcmc_samples::Int64=100,
	num_mcmc_chains::Int64=1,
	use_avg::Bool=true
)
	@assert direction in (:left, :up, :down, :right)

	# make a copy of robot path, and move it
	test_robot = deepcopy(robot_path)
	move!(test_robot, direction)

	# calculate probability of missing
	indicies = pos_to_index(test_robot[end])
	prob_miss = 1.0 - pr_field[(indicies...)]

	# update probability field as if source wasn't found
	test_map = miss_source(test_robot[end], pr_field)
	exp_entropy = 0.0

	r_test = test_robot[end]

	if use_avg
		mean(chain[:, "x₀[1]"])
		x₀_test = [mean(chain[:, "x₀[1]"]), mean(chain[:, "x₀[2]"])]
		R_test = mean(chain[:, "R"])
		c_test = c(r_test, x₀_test, R_test)

		test_data_row = DataFrame(
			"time" => [length(data[:, 1])+1],
			"x [m]" => [r_test],
			"counts" => [c_test]
		)

		test_data = vcat(data, test_data_row)
		test_prob_model = rad_model(test_data)
		test_chain = DataFrame(
			sample(test_prob_model, NUTS(), MCMCSerial(), num_mcmc_samples, num_mcmc_chains)
		)

		P_test = chain_to_P(test_chain)
		return entropy(P_test)

	else
		for row in eachrow(chain)
			x₀_test = [row["x₀[1]"], row["x₀[2]"]]
			R_test = row["R"]
			c_test = c(r_test, x₀_test, R_test)
	
			test_data_row = DataFrame(
				"time" => [length(data[:, 1])+1],
				"x [m]" => [r_test],
				"counts" => [c_test]
			)
	
			test_data = vcat(data, test_data_row)
			test_prob_model = rad_model(test_data)
			test_chain = DataFrame(
				sample(test_prob_model, NUTS(), MCMCSerial(), num_mcmc_samples, num_mcmc_chains)
			)
	
			P_test = chain_to_P(test_chain)
			exp_entropy += entropy(P_test)
		end
		exp_entropy = exp_entropy / length(chain[:, 1])

		return exp_entropy * prob_miss
	end


	
end

# ╔═╡ 8b98d613-bf62-4b2e-9bda-14bbf0de6e99
"""
Given the robot path, returns a tuple of optional directions the robot could travel in next.

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=2.0` - step size of the robot.

"""
function get_next_steps(
	robot_path::Vector{Vector{Float64}}, 
	L::Float64; 
	Δx::Float64=2.0,
	allow_overlap::Bool=false
)
	current_pos = robot_path[end]

	directions = Dict(
        :up    => [0.0, Δx],
        :down  => [0.0, -Δx],
        :left  => [-Δx, 0.0],
        :right => [Δx, 0.0]
    )

	#visited = Set( (pos[1], pos[2]) for pos in robot_path )
	if length(robot_path) > 1
		visited = Set((pos[1], pos[2]) for pos in [robot_path[end-1]])
	else
		visited = ()
	end

	if allow_overlap
		valid_directions = Tuple(
	        dir for (dir, delta) in directions
	        if let new_pos = current_pos .+ delta
	            in_bounds = 0.0 ≤ new_pos[1] ≤ L && 0.0 ≤ new_pos[2] ≤ L
	            in_bounds
	        end
	    )
	else
		valid_directions = Tuple(
	        dir for (dir, delta) in directions
	        if let new_pos = current_pos .+ delta
	            in_bounds = 0.0 ≤ new_pos[1] ≤ L && 0.0 ≤ new_pos[2] ≤ L
	            not_visited = (new_pos[1], new_pos[2]) ∉ visited
	            in_bounds && not_visited
	        end
	    )
	end

	return valid_directions

end

# ╔═╡ 8137f10d-255c-43f6-81c7-37f69e53a2e9
"""
Given the robot path, finds the best next direction the robot to travel using the infotaxis metric.

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `pr_field::Matrix{Float64}` - current posterior for the source location.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
* `data::DataFrame` - current data frame containing sample data.
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=2.0` - step size of the robot.
* `use_avg::Bool=true` - if true, will average the properties from the posterior, if false the algorithm will calculate a posterior from every single sample from the mcmc chain (WARNING, THIS IS VERY EXPENSIVE).
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
"""
function infotaxis(
	robot_path::Vector{Vector{Float64}}, 
	pr_field::Matrix{Float64},
	chain::DataFrame,
	data::DataFrame;
	num_mcmc_samples::Int64=100,
	num_mcmc_chains::Int64=1,
	L::Float64=50.0,
	Δx::Float64=2.0,
	use_avg::Bool=true,
	allow_overlap::Bool=false)

	direction_options = get_next_steps(robot_path, L, allow_overlap=allow_overlap)

	if length(direction_options) < 1 && allow_overlap == true
		@warn "found no viable direction options with overlap allowed, returning nothing"
		return :nothing
	end

	min_entropy = Inf
	best_direction = :nothing
	entropies = Dict(dir => Inf for dir in direction_options)
	

	for direction in direction_options
		exp_entropy = expected_entropy(
			robot_path, 
			direction, 
			pr_field, 
			chain, 
			data,
			num_mcmc_samples=num_mcmc_samples,
			num_mcmc_chains=num_mcmc_chains,
			use_avg=use_avg
	)
		entropies[direction] = exp_entropy
		
		if exp_entropy < min_entropy
			best_direction = direction
			min_entropy = exp_entropy
		end
	end

	if best_direction == :nothing && allow_overlap == false
		return infotaxis(
			robot_path, 
			pr_field,
			chain,
			data,
			num_mcmc_samples=num_mcmc_samples,
			num_mcmc_chains=num_mcmc_chains,
			L=L,
			Δx=Δx,
			use_avg=use_avg,
			allow_overlap=true)
	end

	if best_direction == :nothing
		if all(isinf, values(entropies))
			@warn "all direction options returning infinty, returning random direction choice from $(direction_options) at current location$(robot_path[end])."
			return rand(direction_options)
		end
	end
	
	return best_direction

end

# ╔═╡ a2154322-23de-49a6-9ee7-2e8e33f8d10c
"""
Given the robot path, finds the best next direction the robot to travel using the infotaxis metrix.

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `pr_field::Matrix{Float64}` - current posterior for the source location.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
* `data::DataFrame` - current data frame containing sample data.
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=2.0` - step size of the robot.
* `use_avg::Bool=true` - if true, will average the properties from the posterior, if false the algorithm will calculate a posterior from every single sample from the mcmc chain (WARNING, THIS IS VERY EXPENSIVE).
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
"""
function thompson_sampling(
	robot_path::Vector{Vector{Float64}}, 
	chain::DataFrame;
	L::Float64=50.0,
	Δx::Float64=2.0,
	allow_overlap::Bool=false)

	direction_options = get_next_steps(robot_path, L, allow_overlap=allow_overlap)

	directions = Dict(
        :up    => [0.0, Δx],
        :down  => [0.0, -Δx],
        :left  => [-Δx, 0.0],
        :right => [Δx, 0.0]
    )

	if length(direction_options) < 1 && allow_overlap == true
		@warn "found no viable direction options with overlap allowed, returning nothing"
		return :nothing
	end

	best_direction = :nothing
	greedy_dist = Inf

	#randomly sample from the chain
	rand_θ = chain[rand(1:nrow(chain)), :]
	loc = robot_path[end]

	for direction in direction_options
		new_loc = loc .+ directions[direction]
		dist = norm([rand_θ["x₀[1]"]-new_loc[1], rand_θ["x₀[2]"]-new_loc[2]])
		if dist < greedy_dist
			greedy_dist = dist
			best_direction = direction
		end
	end

	if best_direction == :nothing && allow_overlap == false
		@warn "best direction == nothing, switching to allow overlap"
		return thompson_sampling(
			robot_path, 
			chain,
			L=L,
			Δx=Δx,
			allow_overlap=true)
	end
	
	return best_direction

end

# ╔═╡ 76a9cb27-7cde-44a1-b845-e07cf7a8fa44
"""
Given the robot path, finds the best next direction the robot to travel using the method indicated (infotaxis or thompson)

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `pr_field::Matrix{Float64}` - current posterior for the source location.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
* `data::DataFrame` - current data frame containing sample data.
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=2.0` - step size of the robot.
* `use_avg::Bool=true` - if true, will average the properties from the posterior, if false the algorithm will calculate a posterior from every single sample from the mcmc chain (WARNING, THIS IS VERY EXPENSIVE).
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
* `method::String="infotaxis"` - the method to use for decision making. Must be either `"infotaxis"` or `"thompson"`.
"""
function find_opt_choice(
	robot_path::Vector{Vector{Float64}}, 
	pr_field::Matrix{Float64},
	chain::DataFrame,
	data::DataFrame;
	num_mcmc_samples::Int64=100,
	num_mcmc_chains::Int64=1,
	L::Float64=50.0,
	Δx::Float64=2.0,
	use_avg::Bool=true,
	allow_overlap::Bool=false,
	method::String="infotaxis")

	@assert method == "infotaxis" || method == "thompson" "method must be either infotaxis or thompson: method=$(method) is invalid."

	if method == "infotaxis"
		return infotaxis(
			robot_path, 
			P,
			chain,
			sim_data,
			num_mcmc_samples=num_mcmc_samples,
			num_mcmc_chains=num_mcmc_chains,
			L=L,
			Δx=Δx,
			use_avg=use_avg
		)

	elseif method == "thompson"
		return thompson_sampling(
			robot_path, 
			chain,
			L=L,
			Δx=Δx
		)

	else
		error("invalid method error, please check conditionals as this error code should not be reachable.")
	end


end

# ╔═╡ e278ec3e-c524-48c7-aa27-dd372daea005
"""
TODO:
input should be starting location and a prior. It should check the information gain (entropy reduction) from each possible action and choose the action that reduces entropy the most.
"""
function sim(
	num_steps::Int64; 
	robot_start::Vector{Int64}=[0, 0], 
	num_mcmc_samples::Int64=2000,
	num_mcmc_chains::Int64=1,
	L::Float64=50.0,
	Δx::Float64=2.0,
	x₀::Vector{Float64}=[25.0, 4.0],
	R::Float64=10.0,
	use_avg::Bool=true,
	method::String="thompson",
	save_chains::Bool=false,
	model_params::Union{Nothing, Dict{Any, Any}}=nothing
)

	@assert (method == "infotaxis" || method == "thompson") "method must be either infotaxis or thompson: method=$(method) is invalid."
	
	sim_chains = Dict()

	#times = 1:num_steps
	#xs = robot_start
	r_start = [robot_start[i] * Δx for i=1:2]
	if isnothing(model_params)
		c_start = count_Poisson(r_start, x₀, I, measure=true)
	else
		c_start = sample_model(r_start, model_params, Δx=Δx)
	end
	

	sim_data = DataFrame(
		"time" => [1.0],
		"x [m]" => [r_start],
		"counts" => [c_start]
	)

	robot_path = [r_start]

	for iter = 1:num_steps

		model = rad_model(sim_data)

		
		model_chain = DataFrame(
			sample(model, NUTS(), MCMCSerial(), num_mcmc_samples, num_mcmc_chains)
		)

		if save_chains
			sim_chains[iter] = model_chain
		end
		
		P = chain_to_P(model_chain)

		if norm([robot_path[end][i] - x₀[i] for i=1:2]) < Δx
			@info "Source found at step $(iter), robot at location $(robot_path[end])"
			break
		end
		
		best_direction = find_opt_choice(
			robot_path, 
			P,
			model_chain,
			sim_data,
			num_mcmc_samples=num_mcmc_samples,
			num_mcmc_chains=num_mcmc_chains,
			L=L,
			Δx=Δx,
			use_avg=use_avg,
			method=method
		)

		if best_direction == :nothing
			@warn "iteration $(iter) found best_direction to be :nothing"
			return sim_data
		end
			

		move!(robot_path, best_direction, Δx=Δx)
		if isnothing(model_params)
			c_measurement = count_Poisson(robot_path[end], x₀, I, measure=true)
		else
			c_measurement = sample_model(robot_path[end], model_params, Δx=Δx)
		end
		
		push!(
			sim_data,
			Dict("time" => iter+1.0, 
			"x [m]" => robot_path[end], 
			"counts" => c_measurement
			)
		)
	end

	if save_chains
		return sim_data, sim_chains
	else
		return sim_data
	end
end

# ╔═╡ 17523df5-7d07-4b96-8a06-5c2f0915d96a
simulation_data, simulation_chains = sim(30, method="thompson", save_chains=true, num_mcmc_samples=600, num_mcmc_chains=2)

# ╔═╡ cf110412-747d-44fa-8ab9-991b863eecb3
viz_data(simulation_data, source=x₀, incl_model=true)

# ╔═╡ 474f7e4b-2b95-4d4e-a82a-2d0ab6cffdcf
@bind chain_val PlutoUI.Slider(1:size(simulation_data, 1)-1, show_value=true)

# ╔═╡ 962d552d-9cb2-4a69-9338-5995f7788b96
begin
	#@bind chain_val PlutoUI.Slider(1:size(simulation_data, 1)-1, show_value=true)
	current_chain = simulation_chains[chain_val]
	 viz_chain_data(current_chain, data=simulation_data[1:chain_val, :])
end

# ╔═╡ 139eb9e5-d126-4202-b621-47c38ce1ab93
 viz_posterior(current_chain)

# ╔═╡ d296d31a-d63e-4650-920e-6ab870f4b617
md"""
# Simulation informed work

"""

# ╔═╡ e49b85a4-e52c-48c8-aedc-8e966a5aa8b2
md"""
# TODO

* Implement function to import data from Dr. Yang's mesh modeling software.

* Once data is imported for both simulations with and without obstructions. Create visualization to match the vizualizations provided by Dr. Yang.

* Convert values to counts and place agent and test naive approach by placing in multiple locations. Compare with and without obstructions and compare to 1/r^2 model.
"""

# ╔═╡ 0b8293ac-46c1-41ce-8aaf-53aab6a1a8c1
md"""
## Filename
"""

# ╔═╡ 2b3183d1-f4ca-4549-a43c-b97958308238


# ╔═╡ ce4bea8d-2da4-4832-9aa4-a348cbbe3812
example_data = extract_data(data_files[2])

# ╔═╡ 91a58f53-0d7a-4026-8786-aae78d243c61
example_params = extract_parameters(example_data)

# ╔═╡ 0f840879-7f31-4d31-b8b9-28c6ddac19a8
md"""
## visual
"""

# ╔═╡ 643723ec-a1b5-4ae2-acb6-c87c6b7d63db
viz_model_data(example_params)

# ╔═╡ b25f5d75-9516-405b-89b6-ce685d2112ee
md"""
# sample model
"""

# ╔═╡ 0d60828a-0fb4-404a-b411-010ef464f011
sample_model([250.0, 250.0], example_params)

# ╔═╡ 406a486a-9fb1-4d34-aee2-3e48fa5480a4


# ╔═╡ 8d4cfc03-d13e-43cd-b9f4-b057f7173f21
model_simulation_data, model_simulation_chains = sim(30, method="thompson", save_chains=true, num_mcmc_samples=600, num_mcmc_chains=2, model_params=example_params, Δx=example_params["Δx_y"])

# ╔═╡ b14d282d-de10-4712-9f0f-2ca2b3b0ca3b
example_params

# ╔═╡ 10ed85de-223a-40a1-99bb-f2e6c613a3f8
@bind model_chain_val PlutoUI.Slider(1:size(model_simulation_data, 1)-1, show_value=true)

# ╔═╡ 1e8b7eaa-004e-4c40-82ca-d63d4079360b
begin
	#@bind model_chain_val PlutoUI.Slider(1:size(simulation_data, 1)-1, show_value=true)
	counts_I = I * example_params["γ_matrix"][:, :, 1]
	L_model = example_params["L_xy"] * example_params["Δx_y"]
	current_model_chain = model_simulation_chains[model_chain_val]
	
	 viz_chain_data(current_model_chain, example_params, counts_I, data=model_simulation_data[1:model_chain_val, :], L=L_model, show_source=false)
end

# ╔═╡ Cell order:
# ╠═285d575a-ad5d-401b-a8b1-c5325e1d27e9
# ╠═54b50777-cfd7-43a3-bcc2-be47f117e635
# ╟─52d76437-0d60-4589-996f-461eecf0d45d
# ╟─e5b1369e-0b4b-4da3-9c95-07ceda11b31d
# ╠═064eb92e-5ff0-436a-8a2b-4a233ca4fa42
# ╠═b8d6c195-d639-4438-8cab-4dcd99ea2547
# ╟─82577bab-5ce9-4164-84db-9cfa28b501b0
# ╟─a9c65b66-eb92-4943-91a9-9d0ea6cfa3d3
# ╠═57478c44-578e-4b53-b656-c9b3c208dec4
# ╟─03910612-d3fe-481c-bb70-dd5578bd8258
# ╠═dd357479-64ef-4823-8aba-931323e89aed
# ╟─7fcecc0e-f97c-47f7-98db-0da6d6c1811e
# ╠═e62ba8da-663a-4b58-afe6-910710d7518e
# ╠═19c95a83-670c-4ad6-82a1-5a4b6809f1d4
# ╠═1197e64f-34c2-4892-8da5-3b26ee6e7c2f
# ╟─7278adb5-2da1-4ea1-aa38-d82c23510242
# ╠═63c8b6dd-d12a-42ec-ab98-1a7c6a991dbd
# ╠═7211ea6e-6535-4e22-a2ef-a1994e81d22a
# ╠═1f12fcd1-f962-45e9-8c07-4f42f869d6a0
# ╠═7ed3ac02-2915-4995-8ff4-ea781b283f7f
# ╟─849ef8ce-4562-4353-8ee5-75d28b1ac929
# ╠═e622cacd-c63f-416a-a4ab-71ba9d593cc8
# ╟─8ed5d321-3992-40db-8a2e-85abc3aaeea0
# ╠═6fa37ac5-fbc2-43c0-9d03-2d194e136951
# ╠═0175ede7-b2ab-4ffd-8da1-278120591027
# ╠═bc761684-699f-430b-bcb9-9b03410fa1d5
# ╠═5b9aaaeb-dbb3-4392-a3a1-ccee94d75fed
# ╟─31864185-6eeb-4260-aa77-c3e94e467558
# ╟─015b9f4d-09b8-49f3-bc03-2fd3b972e933
# ╠═126df6ec-9074-4712-b038-9371ebdbc51d
# ╟─bfe17543-7b54-4f52-9679-f723adafdbdd
# ╠═c6c39d74-4620-43df-8eb1-83c436924530
# ╠═3c0f8b63-6276-488c-a376-d5c554a5555d
# ╠═ddc23919-17a7-4c78-86f0-226e4d447dbe
# ╠═50e623c0-49f6-4bb5-9b15-c0632c3a88fd
# ╟─de738002-3e80-4aab-bedb-f08533231ed7
# ╠═deae0547-2d42-4fbc-b3a9-2757fcfecbaa
# ╠═82425768-02ba-4fe3-ab89-9ac95a45e55e
# ╠═9fbe820c-7066-40b5-9617-44ae0913928e
# ╟─3ae4c315-a9fa-48bf-9459-4b7131f5e2eb
# ╟─c6783f2e-d826-490f-93f5-3da7e2717a02
# ╠═1e7e4bad-16a0-40ee-b751-b2f3664f6620
# ╠═e63481a3-a50a-45ae-bb41-9d86c0a2edd0
# ╟─21486862-b3c2-4fcc-98b2-737dcc5211fb
# ╠═2fe974fb-9e0b-4c5c-9a5a-a5c0ce0af065
# ╠═0a39daaa-2c20-471d-bee3-dcc06554cf78
# ╠═ea2dc60f-0ec1-4371-97f5-bf1e90888bcb
# ╟─65d603f4-4ef6-4dff-92c1-d6eef535e67e
# ╠═6da11e28-c276-4f45-a1aa-2c86ab26c85a
# ╠═50830491-6285-4915-b59a-fa5bb7298e51
# ╠═065befd1-f652-4925-b1b2-4e847a3884dd
# ╠═e1303ce3-a8a3-4ac1-8137-52d32bf222e2
# ╠═e7567ef6-edaa-4061-9457-b04895a2fca2
# ╠═bd0a5555-cbe5-42ae-b527-f62cd9eff22f
# ╟─aa72cf61-839d-4707-95c8-0a9230e77d56
# ╠═f4d234f9-70af-4a89-9a57-cbc524ec52b4
# ╠═4bb02313-f48b-463e-a5b6-5b40fba57e81
# ╠═325d565d-ef0e-434a-826a-adb68825f0fd
# ╟─c8f33986-82ee-4d65-ba62-c8e3cf0dc8e9
# ╟─c37e3d82-2320-4278-8b9b-24912a93fd96
# ╟─10fe24bf-0c21-47cc-85c0-7c3d7d77b78b
# ╠═06ecc454-9cd5-432d-bc1c-b269ee3f0794
# ╟─e98ea44e-2da3-48e9-be38-a43c6983ed08
# ╟─14b34270-b47f-4f22-9ba4-db294f2c029c
# ╠═baa90d24-6ab4-4ae8-9565-c2302428e9e7
# ╠═5695ee1e-a532-4a89-bad1-20e859016174
# ╠═5509a7c1-1c91-4dfb-96fc-d5c33a224e73
# ╟─f04d1521-7fb4-4e48-b066-1f56805d18de
# ╠═83052e75-db08-4e0a-8c77-35487c612dae
# ╠═8b98d613-bf62-4b2e-9bda-14bbf0de6e99
# ╠═8137f10d-255c-43f6-81c7-37f69e53a2e9
# ╠═a2154322-23de-49a6-9ee7-2e8e33f8d10c
# ╠═76a9cb27-7cde-44a1-b845-e07cf7a8fa44
# ╠═e278ec3e-c524-48c7-aa27-dd372daea005
# ╠═17523df5-7d07-4b96-8a06-5c2f0915d96a
# ╠═cf110412-747d-44fa-8ab9-991b863eecb3
# ╠═962d552d-9cb2-4a69-9338-5995f7788b96
# ╠═474f7e4b-2b95-4d4e-a82a-2d0ab6cffdcf
# ╠═139eb9e5-d126-4202-b621-47c38ce1ab93
# ╟─d296d31a-d63e-4650-920e-6ab870f4b617
# ╟─e49b85a4-e52c-48c8-aedc-8e966a5aa8b2
# ╟─0b8293ac-46c1-41ce-8aaf-53aab6a1a8c1
# ╠═2b3183d1-f4ca-4549-a43c-b97958308238
# ╠═ce4bea8d-2da4-4832-9aa4-a348cbbe3812
# ╠═91a58f53-0d7a-4026-8786-aae78d243c61
# ╟─0f840879-7f31-4d31-b8b9-28c6ddac19a8
# ╠═643723ec-a1b5-4ae2-acb6-c87c6b7d63db
# ╟─b25f5d75-9516-405b-89b6-ce685d2112ee
# ╠═0d60828a-0fb4-404a-b411-010ef464f011
# ╠═406a486a-9fb1-4d34-aee2-3e48fa5480a4
# ╠═8d4cfc03-d13e-43cd-b9f4-b057f7173f21
# ╠═1e8b7eaa-004e-4c40-82ca-d63d4079360b
# ╠═b14d282d-de10-4712-9f0f-2ca2b3b0ca3b
# ╠═10ed85de-223a-40a1-99bb-f2e6c613a3f8

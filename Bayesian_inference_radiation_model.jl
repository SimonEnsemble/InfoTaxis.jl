### A Pluto.jl notebook ###
# v0.20.8

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
	
	using CairoMakie, LinearAlgebra, Turing, SpecialFunctions, ColorSchemes, DataFrames, StatsBase, PlutoUI, Test, Distributions, Printf, PlutoTeachingTools, JLD2, CSV, DelimitedFiles, LatinHypercubeSampling, Logging, Interpolations
end

# ╔═╡ a03021d8-8de2-4c38-824d-8e0cb571b9f1
begin
	SimulationSpace = include(joinpath("src", "SimulationSpace.jl"))
	RadModelStructs = include(joinpath("src", "RadModelStructs.jl"))
	ExperimentSpace = include(joinpath("src", "ExperimentSpace.jl"))
	LoadData = include(joinpath("src", "LoadData.jl"))
end

# ╔═╡ 4358f533-5944-406f-80a1-c08f99610d5b
Threads.@threads for i=1:10
	print(i)
end

# ╔═╡ 54b50777-cfd7-43a3-bcc2-be47f117e635
TableOfContents()

# ╔═╡ 2d9fc926-9ed0-4a29-8f5c-d57a7b2300fe
md"## QUICK SRC TESTS"

# ╔═╡ 64830991-da9b-4d5d-a0db-3030d9461a8d
begin
	local robot_starts = SimulationSpace.gen_sample_starts(num_samples=2)
	local file = joinpath(joinpath(@__DIR__, "sim_data"), "meshtap")
	local rad_sim = LoadData.import_data(file)

	#= uncomment to run the batch test 
	src_batch_test = SimulationSpace.run_batch(
        rad_sim,
        robot_starts;
        exploring_start=true,
        num_exploring_start_steps=5,
        r_check=60.0,
		num_replicates=1,
        filename="just_some_test"
    )
	=#
end

# ╔═╡ 52d76437-0d60-4589-996f-461eecf0d45d
md"""
# Declarations
"""

# ╔═╡ e5b1369e-0b4b-4da3-9c95-07ceda11b31d
md"## constants"

# ╔═╡ f639c03d-bdc3-43e5-b864-3277bbf02273
@__DIR__

# ╔═╡ 064eb92e-5ff0-436a-8a2b-4a233ca4fa42
begin
	# size of search space
	L = 1000.0 # m, assuming space is square
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
	mCi = 0.10 #100 μCi\

	I = mCi * 3.7 * 10^7 * P_γ # 1mCi = 3.7*10^7 Bq
	#counts/gamma - multiply this by the value normalized to #of photons

	# colors
	colormap = reverse(vcat([ColorSchemes.hot[i] for i in 0.0:0.02:1], ColorSchemes.batlow[0.0]))

	# Locate files for which data needs to be extracted
	dir_name = "sim_data"
	data_dir = joinpath(pwd(), dir_name)
	data_files = [joinpath(data_dir, file) for file in readdir(data_dir)]

	# Turing parameters
	I_max = 1e10 #emmissions/s
	I_min = 1e6

	# robot/environment parameters
	r_velocity = 5.0 #m/s
	λ_background = 0.5 #Poisson distr lambda val for noise

end

# ╔═╡ dd15ee55-76cd-4d56-b4a6-aa46212c176b
data_files

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
	#= I set this up just in case, but I don't expect to use this...
	if we do use this, I'll probably have to use another library with inpolygon() function that determins if a point lies inside a polygon.
	struct Polygon <: Obstruction
	    vertices::Vector{Tuple{Float64, Float64}}
	end
	=#
end

# ╔═╡ 03910612-d3fe-481c-bb70-dd5578bd8258
md"### rad sim"

# ╔═╡ dd357479-64ef-4823-8aba-931323e89aed
struct RadSim
    γ_matrix::Vector{Matrix{Float64}} #vector of matrices of gamma values. Multiply by Intensity to get counts/s, each entry of the vector represents a z-slice
    Δxy::Float64 #m, the size of each step in the x,y plane of the γ_matrix
    Δz::Float64 #m, the size of each step in the z plane
    Lxy::Float64 #m, the size of the x, y plane... assume square
    Lz::Float64 #m, the size of the z plane
	x₀::Vector{Float64} #the coordinates of the source
end

# ╔═╡ aca647fb-d8c4-4966-a641-3b361295f1e2
md"### environment"

# ╔═╡ 2eba07e7-7379-4b2d-bad3-a6f1d52b676e
struct Environment
	env::Matrix{Int64} #unaltered mapped environment from vacuum robot
	masked_env::Matrix{Int64} #masked environment after flood fill algorith applied
	grid::Array{Union{Bool, Int64}, 3} #array of patrol grid [x coord, y coord, is_obstructed()]
	Δ::Float64 #grid spacing
end

# ╔═╡ 7fcecc0e-f97c-47f7-98db-0da6d6c1811e
md"""
# Data Import
"""

# ╔═╡ 9ac1e08a-cd89-4afb-92d9-2bd973f06aaa
md"## save CSV from python"

# ╔═╡ e80af66e-54ab-4661-bd49-89328c17e3d4
"""
Function to read csv file generated by the rad teams python script. This will take the python CSV file, remove extraneous characters, turn it into a matrix of 1's and 0's and orient it properly. The 1's in the matrix represent known obstructions or walls and the 0's represent empty space. Each index of the matrix represents a centimeter as interpreted by the vacuum robot.

# arguments
* `path::String` - the path to the csv file we want to convert to a matrix.
# returns
* `Matrix{Int}` – A square matrix of `1`s and `0`s where:
  - `1` represents an obstruction (e.g., a wall or object),
  - `0` represents free space.
The matrix is reoriented such that the vertical axis is flipped to match the expected coordinate system for the environment, with each index corresponding to a 1 cm × 1 cm grid cell.
"""
function parse_numpy_csv_file(path::String)
    raw = read(path, String)

    stripped = replace(raw, ['[', ']', '"', '\n', '.'] => "")
    tokens = split(strip(stripped))
    values = round.(Int, parse.(Float64, tokens))

    #check if it's a perfect square
    n = length(values)
    ncols = Int(round(sqrt(n)))
    @assert ncols^2 == n "Matrix is not square: total elements = $n, but sqrt = $ncols"

    #reshape into square matrix
	square_matrix = reshape(values, ncols, ncols)

	#flip x axis
	#square_matrix_flipped = square_matrix[end:-1:1, :]
	square_matrix_flipped = square_matrix[:, end:-1:1]
	
    return square_matrix_flipped
end

# ╔═╡ 2c959785-6d71-49c6-921a-16e74cb3b43e
vac_environment = parse_numpy_csv_file(joinpath("csv", "Walls.csv"))

# ╔═╡ 0fc694a6-f4cf-478d-bd68-9af5f7f4f5b8
heatmap(vac_environment)

# ╔═╡ 181c27f4-6830-4c4d-9392-3237564e6cb1
md"## obstruction data"

# ╔═╡ 7c44611e-2442-4dca-9624-e18676e0f67c
md"""
#### This obstruction data was provided in an email by the rad team.
"""

# ╔═╡ 52814746-ae35-4ffa-9be0-66854a4d96bf
"""
Returns a vector of obstruction objects used in our example simulation.
"""
function example_obstructions()
	#rectangular block x(150-350) y(650-750) z(0-200)
	wide_rect = Rectangle(
		(150.0 + (350.0-150.0)/2, 650.0 + (750.0-650.0)/2),
		350.0-150.0,
		750.0-650.0
	)
	#rectangular block x(450-550) y(450-550) z(0-200)
	square = Rectangle(
		(450.0 + (550.0-450.0)/2, 450.0 + (550.0-450.0)/2),
		550.0-450.0,
		550.0-450.0
	)
	#cylinder bottom (750,350,0) height = 200, radius = 50
	cylinder = Circle(
		(750.0, 350.0),
		50.0
	)
	obstructions = [wide_rect, square, cylinder]
	return obstructions
end

# ╔═╡ a092d5de-8828-4fa6-8ef5-fb0838cc0887
obstructions = example_obstructions()

# ╔═╡ 19c95a83-670c-4ad6-82a1-5a4b6809f1d4
"""
Helper function used by `import_data()` to convert the dictionary read from the text file into dictionary containing the components necessary to make a RadSim struct.

# arguments
* `data::Dict` - the dictionary made within `import_data()` that contains keys:
`"y_bin_bounds"`, `"Δxy"`, `"z_bin_bounds"`, `"Δz"`, `"Lxy"`, `"Lz"`, `"energy_field_data"`
"""
function get_matrix_params(data::Dict)

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
"""
Reads the lines of the simulation data file provided by the radiation team at path: `data_file_path::String` and returns a RadSim data struct.

# arguments
* `data_file_path::String` - the path to the simulation data file.
# keyword arguments
* `x₀::Vector{Float64}=[250.0, 250.0]` - the coordinates of the source, this will be stored in the data structure returned by `import_data`.
# returns
* `RadSim` – A radiation simulation data structure containing:
  - `γ_matrix::Vector{Matrix{Float64}}` – A list of 2D matrices (one per z-slice) representing radiation interaction data.
  - `Δxy::Float64` – Spatial resolution in the x-y plane.
  - `Δz::Float64` – Spatial resolution along the z-axis.
  - `Lxy::Float64` – Side length (in meters) of the square x-y plane.
  - `Lz::Float64` – Total height (in meters) along the z-axis.
  - `x₀::Vector{Float64}` – The specified source location used in simulation.
"""
function import_data(data_file_path::String; x₀::Vector{Float64}=[250.0, 250.0])
	#ensure the input is a file.
	@assert isfile(data_file_path) "$(data_file_path) is not a file!!!"

	#flags used to indicate sections of the data file.
	flags = Dict(
		"tally_bin_bounds" => false,
		"x_dir" => false,
		"y_dir" => false,
		"z_dir" => false,
		"energy_grid" => false
	)

	#set up a dictionary to hold the data
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

	#read the rad sim file line by line
	open(data_file_path) do f
		while !eof(f)
			#read the line
			f_line = lowercase(readline(f))
			# replace double+ spaces with single spaces
			s = replace(f_line, r"\s{2,}" => " ")

			#check for first line of data
			#there are two main sections, "tally bin boundaries" and "energy x y"
			if contains(s, "tally bin boundaries")
				flags["tally_bin_bounds"] = true
				continue
			elseif contains(s, "energy x y")
				flags["energy_grid"] = true
				continue
			end

			#tally bin bounds section
			#within this section we use the x, y, z direction flags
			if flags["tally_bin_bounds"]
				#if we reach the next section, turn off the flag and continue
				if contains(s, "energy bin boundaries")
					flags["tally_bin_bounds"] = false
					continue
				end
   				for dir in ["x", "y", "z"]
					#check if the beginning of a new direction
        			if occursin("$dir direction", s)
            			flags["x_dir"] = flags["y_dir"] = flags["z_dir"] = false
           				flags["$(dir)_dir"] = true
						
						values = parse.(Float64, split(s)[3:end])
						data["$(dir)_bin_bounds"] = values
					#if not the beginning of a new direction, collect data
					elseif !(occursin("direction", s))
						#only collect data for the current flagged direction
						if flags["$(dir)_dir"]
							values = parse.(Float64, split(s))
							data["$(dir)_bin_bounds"] = vcat(data["$(dir)_bin_bounds"], values)
						end
					end
				end
			end
			#for engery_grid, just push new lines to the dataframe
			if flags["energy_grid"]
				values = parse.(Float64, split(s))
				push!(data["energy_field_data"], values)
			end
		end
	end

	#convert data into a matrix
	matrix_params = get_matrix_params(data)

	#generate and return a rad_sim struct
	rad_sim = RadSim(
		[matrix_params["γ_matrix"][:, :, i] for i in 1:size(matrix_params["γ_matrix"], 3)],
		matrix_params["Δxy"],
		matrix_params["Δz"],
		matrix_params["Lxy"],
		matrix_params["Lz"],
		x₀
	)
	return rad_sim
end

# ╔═╡ 8ffaf344-1c74-48c8-a116-8c937322cd6e
md"""
## import radiation simulation data
"""

# ╔═╡ 1197e64f-34c2-4892-8da5-3b26ee6e7c2f
begin
	num_sims = length(data_files)
	rad_sim_data = [import_data(data_files[i]) for i=1:num_sims]

	test_rad_sim = rad_sim_data[1]
	test_rad_sim_obstructed = rad_sim_data[2]
end

# ╔═╡ 7278adb5-2da1-4ea1-aa38-d82c23510242
md"""
## `Visualize` - Imported Data
"""

# ╔═╡ 173feaf5-cbfa-4e94-8de5-1a5311cdf14e
"""
Obstruction visualization helper function.

# arguments
* `ax` - Cairo Makie axis.
* `obstructions::Vector{Obstruction}` - vector containing Obstruction objects to be visualized.
"""
function viz_obstructions!(ax, obstructions::Vector{Obstruction})
	for obs in obstructions
		if obs isa Rectangle
			cx, cy = obs.center
			w2, h2 = obs.width / 2, obs.height / 2
			rect_vertices = [
				(cx - w2, cy - h2),
				(cx + w2, cy - h2),
				(cx + w2, cy + h2),
				(cx - w2, cy + h2),
			]
			poly!(ax, rect_vertices, color=ColorSchemes.bamako10[1], strokewidth=1.0, transparency=true)
		elseif obs isa Circle
			θ = range(0, 2π; length=100)
			cx, cy = obs.center
			r = obs.radius
			xs = cx .+ r .* cos.(θ)
			ys = cy .+ r .* sin.(θ)
			poly!(ax, xs, ys, color=ColorSchemes.bamako10[1])
		end
	end
end

# ╔═╡ 7211ea6e-6535-4e22-a2ef-a1994e81d22a
"""
Radiation simulation RadSim, visual helper function.

# arguments
* `ax` - Cairo Makie axis.
* `rad_sim::RadSim` - radiation simulation struct.
# keyword arguments
* `z_slice::Int64=1` - change to change the z slice of the data, unless the data has more than 2-dimensional data, just keep as 1.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - vector containing Obstruction objects to be visualized.
"""
function viz_model_data!(
	ax, 
	rad_sim::RadSim; 
	z_slice::Int64=1, 	
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing
)
	#set up a log scale
	scale = ReversibleScale(
	    x -> log10(x + 1),   # forward: avoids log(0)
	    x -> 10^x - 1        # inverse
	)

	# x and y values 
	xs = ys = [val for val in 0:rad_sim.Δxy:rad_sim.Lxy]
	#counts
	counts_I = I * rad_sim.γ_matrix[z_slice]

	hm = heatmap!(ax, xs, ys, counts_I, colormap=colormap, colorscale=scale)

	if !isnothing(obstructions)
		viz_obstructions!(ax, obstructions)
	end
	
	return hm
end

# ╔═╡ 63c8b6dd-d12a-42ec-ab98-1a7c6a991dbd
"""
Visualize simulated radiation data provided by the radiation team.

# arguments
* `rad_sim::RadSim` - the radiation simulation data structure after being imported.
# keyword arguments
* `z_slice::Int64=1` - change to change the z slice of the data, unless the data has more than 2-dimensional data, just keep as 1.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - optional obstruction data, if provided obstructions will be visualized as teal blocks.
"""
function viz_model_data(
	rad_sim::RadSim; 
	z_slice::Int64=1, 
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing
)
	fig = Figure()
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x₁", 
	    ylabel="x₂"
	)

	#convert normalized gammas to counts
	counts_I = I * rad_sim.γ_matrix[z_slice]
	
	hm = viz_model_data!(ax, rad_sim, obstructions=obstructions)

	#establish logarithmic colorbar tick values
	colorbar_tick_values = [10.0^e for e in range(0, log10(maximum(counts_I)), length=6)]
	colorbar_tick_values[1] = 0.0
	colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]

	Colorbar(fig[1, 2], hm, label = "counts / s", ticks = (colorbar_tick_values, colorbar_tick_labels))
	
	fig
end

# ╔═╡ 1f12fcd1-f962-45e9-8c07-4f42f869d6a0
viz_model_data(test_rad_sim)

# ╔═╡ bbeed200-3315-4546-9540-16864843f178
viz_model_data(test_rad_sim_obstructed, obstructions=obstructions)

# ╔═╡ 3ee48a88-3b6a-4995-814b-507679065ff4
md"# Create Grid for Vaacuum"

# ╔═╡ 6fdb31a6-f287-40a4-8c21-2ef92ff90a99
"""
Identifies points that are too close (i.e. within radius) of a wall. This is being used to thicken walls for the flood fill algorithm.

# arguments
* `env_matrix::Matrix{Int}` - the environment matrix generated by the vacuum robot containing 0's for opent space and 1's for obstructions.
* `radius::Int` - the degree to which we want to thicken the wall, for a value of 0 this will simply return the original env_matrix. Larger values will create thicker walls.
"""
function clearance_mask(env_matrix::Matrix{Int}, radius::Int)
    h, w = size(env_matrix)
    mask = falses(h, w)

    for y in 1:h, x in 1:w
        for dy in -radius:radius, dx in -radius:radius
            yy, xx = y + dy, x + dx
			#check if the current grid location is too close to an obstacle
            if xx ≥ 1 && xx ≤ w && yy ≥ 1 && yy ≤ h && env_matrix[yy, xx] == 1
                mask[y, x] = true
                break
            end
        end
    end

    return mask
end


# ╔═╡ ca763f28-d2b2-4eac-ae6c-90f74e3c42e7
"""
Breadth first search flood fill (paint bucket) algorithm. Finds adjacent locations that are accessible from a seed location and returns a new matrix excluding locations that cannot be accessed from the seed by traveling adjacently.

# arguments
* `env::Matrix{Int}` - Matrix containing only 0's and 1's, where a 0 represents open space and a 1 represents an obstruction.
* `seed::Tuple{Int, Int}` - The starting location (indicies) for the paint bucket, this index must point to a 0 such that `env[seed]==0`.
# keyword arguments
* `clearance_radius::Int=5` - The thickness of walls, larger values will ensure small openings are blocked, set to 0 to not use a mask at all.
"""
function flood_fill(
	env::Matrix{Int}, 
	seed::Tuple{Int, Int}; 
	clearance_radius::Int=5
)
    h, w = size(env)
    buffer_zone = clearance_mask(env, clearance_radius)
    visited = falses(h, w)

	#check seed validity
    x₀, y₀ = seed
    if env[y₀, x₀] != 0 || buffer_zone[y₀, x₀]
        error("Seed is not in valid, clear space")
    end

	#initialize q as the seed location
    q = [(x₀, y₀)]

	#flood fill algorithm, builds out from the seed until it runs into the buffer_zone
    while !isempty(q)
        (x, y) = pop!(q)

        if x < 1 || x > w || y < 1 || y > h
            continue
        elseif visited[y, x] || env[y, x] != 0 || buffer_zone[y, x]
            continue
        end

        visited[y, x] = true
        append!(q, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    end

	#remake environment as 0's and 1's
    new_env = copy(env)
    for y in 1:h, x in 1:w
        if env[y, x] == 0 && !visited[y, x]
            new_env[y, x] = 1
        end
    end

    return new_env
end


# ╔═╡ 00f909f7-86fd-4d8e-8db2-6a587ba5f12d
begin
	environment_masked = flood_fill(vac_environment, (250, 250), clearance_radius=5)
	heatmap(environment_masked)
end

# ╔═╡ e2387282-0c5e-4115-9868-738c864ce41b
heatmap(vac_environment)

# ╔═╡ 5e7dc22c-dc2d-41b0-bb3e-5583a5b79bdd
"""
Generates a matrix of locations for a robot to explore of user defined coarseness.

# arguments
* `environment::Matrix{Int}` - Matrix containing only 0's and 1's, where a 0 represents open space and a 1 represents an obstruction.
* `step::Int` - The size of each step in the robot exploration matrix. For example, a step of 10 means each index in the robot matrix is 10 times further apart then the original obstruction map.
# keyword arguments
* `mask_outside::Bool=true` - set to true to use flood fill algorithm (paint bucket) to try and remove inaccessible sections of the map.
* `seed::Tuple{Int, Int}=(250,250)` - The starting location (indicies) for the paint bucket, this index must point to a 0 such that `env[seed]==0`.
* `clearance_radius::Int=5` - adjusts the clearance radius of the mask, i.e. thickens the walls.
# returns
* `Environment` – A structured environment object containing:
  - `env::Matrix{Int}` – The original input environment matrix.
  - `masked_env::Matrix{Int}` – The environment after optional flood fill masking (if `mask_outside=true`).
  - `grid::Array{Union{Int, Bool}, 3}` – A 3D array representing the robot's coarser sampling grid:
    - `grid[:, :, 1]` – x-coordinates in the original environment,
    - `grid[:, :, 2]` – y-coordinates in the original environment,
    - `grid[:, :, 3]` – boolean values indicating accessibility at each coarse location.
  - `Δ::Float64` – The spatial resolution (in meters) corresponding to `step`.

"""
function generate_robot_grid_matrix(
	environment::Matrix{Int}, 
	step::Int; 
	mask_outside::Bool=true, 
	seed::Tuple{Int, Int}=(250,250), 
	clearance_radius::Int=5
)
    h, w = size(environment)

    # Optionally apply flood fill to exclude disconnected regions
    if mask_outside
        environment_masked = Int.(flood_fill(environment, 
									  seed, 
									  clearance_radius=clearance_radius
									 )
						  )
	else
		environment_masked = environment
    end
    grid_rows = cld(h, step)
    grid_cols = cld(w, step)

    #grid is made of x, y coordinates of original robot space and boolean to represent accessibility
    grid = Array{Union{Int, Bool}}(undef, grid_rows, grid_cols, 3)

    for i in 1:grid_rows
        for j in 1:grid_cols
            y = min((i - 1) * step + 1, h)
            x = min((j - 1) * step + 1, w)
            valid = environment_masked[y, x] == 0
            grid[i, j, 1] = x
            grid[i, j, 2] = y
            grid[i, j, 3] = valid
        end
    end

	#store everything in data struct
	env = Environment(
		environment,
		environment_masked,
		grid,
		step*1.0
	)

    return env
end

# ╔═╡ 4bda387f-4130-419c-b9a5-73ffdcc184f9
grid_env =  generate_robot_grid_matrix(vac_environment, 10)

# ╔═╡ 560de084-b20b-45d7-816a-c0815f398e6d
md"""
## `Visualize` - Rad source search grid
"""

# ╔═╡ 6d4e86ae-701c-443e-9d05-0cc123adbc29
grid_env

# ╔═╡ b48a8a94-7791-4e10-9a6b-1c6f2eca7968
function resize_env_mask_to_posterior(env_mask, post_n::Int)
    env_n = size(env_mask, 1)

    # Convert Bool mask to Float for interpolation
    float_mask = Float64.(env_mask)

    itp = interpolate(float_mask, BSpline(Linear()))
    scale_itp = scale(itp, range(1, env_n, length=env_n), range(1, env_n, length=env_n))

    # Evaluate at evenly spaced points to match posterior resolution
    query_pts = range(1, env_n, length=post_n)
    resized_mask = [scale_itp[x, y] ≥ 0.5 for x in query_pts, y in query_pts]

    return resized_mask
end

# ╔═╡ b63d1384-02ef-45c6-80c8-fdfd54ce6804
grid_env.grid

# ╔═╡ 849ef8ce-4562-4353-8ee5-75d28b1ac929
md"# Analytical (Poisson) Model"

# ╔═╡ e622cacd-c63f-416a-a4ab-71ba9d593cc8
"""
Generates a Poisson distribution based on position, source position and source strength.

# arguments
* `x::Vector{Float64}` - Measurement/true value location.
* `x₀::Vector{Float64}` - source location.
* `I::Float64` - source strength in Bq.
# returns
* `Poisson` – A Poisson distribution object representing the expected radiation counts at location `x`, based on the inverse square law and exponential attenuation from the source at `x₀`. If the computed rate `λ` is invalid (e.g., `NaN` or negative), a zero-rate Poisson distribution is returned instead.

"""
function count_Poisson(x::Vector{Float64}, x₀, I)
	distance = norm(x₀ .- x)
	attenuation = attenuation_constant(x, x₀)
	λ = I * Δt * ϵ * (A / (4π * distance^2)) * exp(-attenuation)

	#this piece became necessary as NAN or negative values were being tested by the optimizer causing errors
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
"""
`viz_c_analytical` helper function, Visualizes a field of radiation strength values as expected counts per second using an analytical Poisson model.


# arguments
* `ax` - Cairo Makie axis.
* `color_scale::ReversibleScale` - the non-linear color scale.
# keyword arguments
* `fig_size::Int=500` - resolution control.
* `L::Float64=1000.0` - size of the grid space.
* `x₀::Vector{Float64}=[251.0, 251.0]` - source location. This is used to calculate strength.
* `I::Float64=1.16365e10` - source strength.
* `source::Union{Nothing, Vector{Float64}}=x₀` - source location. Set to nothing to remove the scatter plot visual of the source.
* `scale_max::Float64=1e5` - the max source strength for the color bar.
"""
function viz_c_analytical!(
	ax, color_scale::ReversibleScale; 
	fig_size::Int=500, 
	L::Float64=1000.0, 
	x₀::Vector{Float64}=[251.0, 251.0], 
	I::Float64=1.16365e10, 
	source::Union{Nothing, Vector{Float64}}=x₀, 
	scale_max::Float64=1e5
)
	#colormap = reverse([ColorSchemes.hot[i] for i in 0.0:0.05:1])

	xs = collect(0.0:Δx:L)
	counts = [mean(count_Poisson([x₁, x₂], x₀, I)) for x₁ in xs, x₂ in xs] # counts

	hm = heatmap!(ax, xs, xs, counts, colormap=colormap, colorscale = color_scale, colorrange=(0, scale_max))

	if ! isnothing(source)
		scatter!(ax, [source[1]], [source[2]], color="red", marker=:xcross, markersize=10, label="source", strokewidth=1)
		axislegend(ax, position=:rb)
	end

	return hm, counts
end

# ╔═╡ 6fa37ac5-fbc2-43c0-9d03-2d194e136951
"""
Visualizes a field of radiation strength values as expected counts per second using an analytical Poisson model.

# keyword arguments
* `fig_size::Int=500` - resolution control.
* `L::Float64=1000.0` - size of the grid space.
* `x₀::Vector{Float64}=[251.0, 251.0]` - source location. This is used to calculate strength.
* `I::Float64=1.16365e10` - source strength.
* `source::Union{Nothing, Vector{Float64}}=x₀` - source location. Set to nothing to remove the scatter plot visual of the source.
* `scale_max::Float64=1e5` - the max source strength for the color bar.
"""
function viz_c_analytical(; 
	fig_size::Int=500, 
  	L::Float64=1000.0, 
  	x₀::Vector{Float64}=[251.0, 251.0], 
  	I::Float64=1.16365e10, 
  	source::Union{Nothing, Vector{Float64}}=x₀, scale_max::Float64=1e5
 )
	
	fig = Figure(size = (fig_size, fig_size))
	ax  = Axis(
	    fig[1, 1], 
	    aspect=DataAspect(), 
	    xlabel="x₁", 
	    ylabel="x₂",
		title="Expected Radiation Field (Poisson Model)"
	)

	scale = ReversibleScale(
	    x -> log10(x + 1),   # forward: avoids log(0)
	    x -> 10^x - 1        # inverse
	)

	hm, _ = viz_c_analytical!(ax, scale, fig_size=fig_size, L=L, x₀=x₀, I=I, source=source, scale_max=scale_max)

	colorbar_tick_values = [10.0^e for e in range(0, log10(scale_max), length=6)]
	colorbar_tick_values[1] = 0.0
	colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]
	

	Colorbar(fig[1, 2], hm, label = "counts / s", ticks = (colorbar_tick_values, colorbar_tick_labels))
	
	fig
end

# ╔═╡ 5b9aaaeb-dbb3-4392-a3a1-ccee94d75fed
viz_c_analytical(scale_max = 1.0*10^6)

# ╔═╡ 31864185-6eeb-4260-aa77-c3e94e467558
md"# Simulate Movement"

# ╔═╡ 015b9f4d-09b8-49f3-bc03-2fd3b972e933
md"## sample environment"

# ╔═╡ bfe17543-7b54-4f52-9679-f723adafdbdd
md"## movement"

# ╔═╡ 83052e75-db08-4e0a-8c77-35487c612dae
"""
Given the grid spacing, provides an index given the provided position vector.

# arguments
* `pos::Vector{Float64}` - current position for which you want the corresponding index.
# keyword arguments
* `Δx::Float64=10.0` - grid spacing.
# returns
* `Tuple{Int, Int}` – A tuple `(i, j)` representing the discrete grid indices corresponding to the input position `pos`. The indices are 1-based and computed by flooring the position divided by the grid spacing `Δx`. This maps continuous coordinates to matrix-style indexing.
"""
function pos_to_index(pos::AbstractVector{<:Real}; Δx::Real=10.0)
    x₁ = Int(floor((pos[1]) / Δx)) + 1
    x₂ = Int(floor((pos[2]) / Δx)) + 1
    return (x₁, x₂)
end

# ╔═╡ 126df6ec-9074-4712-b038-9371ebdbc51d
"""
Given the current position and radiation simulation model, samples the model by pulling the value from the radiation simulation and adding some noise.

# arguments
* `x::Vector{Float64}` - current position for which you are sampling the model.
* `rad_sim` - the radiation simulation RadSim struct containing the simulation data.
# keyword arguments
* `I::Float64=I` - source strength.
* `Δx::Float64=Δx` - grid spacing.
* `z_index::Int=1` - 1 is the ground floor index of the set of 2-D simulation slices.
"""
function sample_model(x::Vector{Float64}, rad_sim::RadSim; I::Float64=I, Δx::Float64=Δx, z_index::Int=1)
	counts_I = I * rad_sim.γ_matrix[z_index]
	@assert count([round(Int, x[i] / Δx) <= size(counts_I, i) && x[i] >= 0.0 for i=1:2]) == 2 "r coordinate values outside of domain"

	#add background noise
	noise = rand(Poisson(λ_background)) * rand([-1, 1])

	#get index from position
	indicies = pos_to_index(x)
	measurement = counts_I[indicies[1] , indicies[2] , z_index] + noise
	measurement = max(measurement, 0)

	return round(Int, measurement)
end

# ╔═╡ c6c39d74-4620-43df-8eb1-83c436924530
"""
Converts a direction to a vector represnetation of movement.

# arguments
* `direction::Symbol` - :up, :down, :left, or :right
# keyword arguments
* `Δx::Float64=Δx` - grid spacing value.
# returns
* `Vector{Float64}` – A 2D vector representing the change in position associated with the given direction. The vector has the form `[Δx, Δy]`, where the magnitude is determined by the `Δx` grid spacing. For example, `:up` returns `[0.0, Δx]`.
"""
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
"""
Moves the robot one step in the specified direction according to the grid spacing `Δx`.

This function appends a new position to the robot's path based on the direction and step size. It does not check for obstructions or boundary limits.

# arguments
* `robot_path::Vector{Vector{Float64}}` – The robot's current path, where the last element is the current position.
* `direction::Symbol` – Direction to move. Must be one of `:up`, `:down`, `:left`, or `:right`.

# keyword arguments
* `Δx::Float64=Δx` – The grid spacing value (movement step size).

# modifies
* `robot_path` – Updated in-place with one additional position in the given direction.
"""
function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol; Δx::Float64=Δx)
	Δ = get_Δ(direction, Δx=Δx)
	push!(robot_path, robot_path[end] + Δ)
end

# ╔═╡ de738002-3e80-4aab-bedb-f08533231ed7
md"## `Visualize` - Movement and Measurement"

# ╔═╡ 82425768-02ba-4fe3-ab89-9ac95a45e55e
"""
`viz_path!` helper function, visualizes the robot’s movement and collected radiation data on a 2D axis.

Each data point is represented as a scatter marker colored by the measured radiation count rate, with optional line segments showing the robot's traversal path. Color and size gradients are used to enhance the visual progression of the path over time.

# arguments
* `ax` – A `CairoMakie.Axis` object to draw the path and measurements on.
* `path_data::DataFrame` – DataFrame containing at least `"x [m]"` (a vector of robot positions) and `"counts"` (scalar values representing measured radiation) for each step.

# keyword arguments
* `scale_max::Real=1e6` – Maximum value for the color scale mapping measured radiation counts.
* `show_lines::Bool=true` – If `true`, draws connecting line segments between consecutive robot positions to show the path trajectory.

# returns
* A `Makie.scatterplot` object representing the colored data points.
"""
function viz_path!(
	ax, 
	path_data::DataFrame; 
	scale_max::Real=1e6, 
	show_lines::Bool=true
)

	
	positions = [(row["x [m]"][1], row["x [m]"][2]) for row in eachrow(path_data)]
	if show_lines
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

	return sc
end

# ╔═╡ 50832f87-c7eb-4418-9864-0f807a16e7a7
md"## Spiral movement"

# ╔═╡ 7d0e24e2-de5b-448c-8884-4d407ead1319
mutable struct SpiralController
    pos::Vector{Float64}
    directions::Vector{Symbol}
    dir_idx::Int
    step_size::Int
    step_increment::Int
    leg_count::Int 
end

# ╔═╡ 22652624-e2b7-48e9-bfa4-8a9473568f9d
"""
Initialize a spiral controller.

# arguments
* `start_pos` - the current location (where the spiral begins)
# keyword arguments
* `step_init::Int=2` - the size of the initial step
* `step_incr::Int=2` - the number of steps by which the spiral will incrementally increase
"""
function init_spiral(start_pos::Vector{Float64}; step_init::Int=2, step_incr::Int=2)
    return SpiralController(
        start_pos,
        [:right, :up, :left, :down],
        1,
        step_init,
        step_incr,
        0
    )
end

# ╔═╡ 3ae4c315-a9fa-48bf-9459-4b7131f5e2eb
md"# Turing MCMC"

# ╔═╡ c6783f2e-d826-490f-93f5-3da7e2717a02
md"## naive rad model"

# ╔═╡ 1e7e4bad-16a0-40ee-b751-b2f3664f6620
"""
`rad_model` defines a probabilistic model for inferring the source location and intensity of a radiation emitter from count data.

This model assumes that measured counts at each location follow a Poisson distribution whose expected value is determined by the distance to a latent source and the source intensity. The source location is modeled as a 2D uniform distribution over a square domain, and the source intensity is drawn from a uniform prior bounded by `I_min` and `I_max`.

# arguments
* `data::DataFrame` – A DataFrame containing at least:
  - `"x [m]"`: a vector of 2D spatial coordinates where measurements were taken.
  - `"counts"`: the corresponding measured count data at those locations.

# keyword arguments
* `L_min::Float64=0.0` – Minimum bound of the spatial domain for the source location prior.
* `L_max::Float64=L` – Maximum bound of the spatial domain for the source location prior.

# returns
* A `Turing.Model` object which can be used for sampling the posterior of the source parameters.
"""
@model function rad_model(data; L_min::Float64=0.0, L_max::Float64=L, environment=nothing)
	# source location
    x₀ ~ filldist(Uniform(L_min, L_max), 2)

	if !isnothing(environment)
	    # soft rejection for inaccessible locations
	    idx = pos_to_index(x₀; Δx=environment.Δ)
	    is_valid = 1 ≤ idx[1] ≤ size(environment.grid, 1) &&
	               1 ≤ idx[2] ≤ size(environment.grid, 2) &&
	               environment.grid[idx[1], idx[2], 3] == true
	
	    Turing.@addlogprob! is_valid ? 0.0 : -Inf
	end
	
	# source strength
	I ~ Uniform(I_min, I_max)

    for i in 1:nrow(data)
        data[i, "counts"] ~ count_Poisson(data[i, "x [m]"], x₀, I)
    end

    return nothing
end

# ╔═╡ 21486862-b3c2-4fcc-98b2-737dcc5211fb
md"## `Visualize` - Turing chain"

# ╔═╡ 2b656540-9228-4d8a-9db1-1c0bec3d33f3
grid_env

# ╔═╡ a8281a0d-2b9c-40d7-802a-ca434ba602f9
size(grid_env.grid, 1)

# ╔═╡ 9dd28728-f8de-43bb-8e0c-e7384f924adc
grid_env

# ╔═╡ 65d603f4-4ef6-4dff-92c1-d6eef535e67e
md"## `Visualize` - Turing chain heatmap"

# ╔═╡ 10c62123-4ae4-4688-87e1-9b0397c75e88
function chain_to_P(chain::DataFrame; Δx::Float64=Δx, L::Float64=L)
    nbins = floor(Int, L / Δx)
    x_edges = range(0, stop=L, length=nbins + 1)
    hist = fit(Histogram, (chain[:, "x₀[1]"], chain[:, "x₀[2]"]), (x_edges, x_edges))
    return hist.weights / sum(hist.weights)
end

# ╔═╡ 8853858d-1d47-40c9-94a4-c912ad00af5d
function edges_to_centers(x_edges)
    return [(x_edges[i] + x_edges[i+1]) / 2 for i in 1:length(x_edges)-1]
end

# ╔═╡ 0a39daaa-2c20-471d-bee3-dcc06554cf78
function viz_chain_data!(
	ax, 
	chain::DataFrame; 
	show_source::Bool=true,
	src_size::Float64=15.0,
	show_as_heatmap::Bool=false,
	Δx::Float64=Δx, 
	L::Float64=L,
	show_legend::Bool=true,
	environment=nothing
)
	#if environment is provided, extract Δ and compute L
	if !isnothing(environment)
		Δx = environment.Δ
		L = Δx * size(environment.grid, 1)
	end
	

	if show_as_heatmap
		nbins = floor(Int, L / Δx)
	    x_edges = range(0, stop=L, length=nbins + 1)
	    x_centers = edges_to_centers(x_edges)
	    P = chain_to_P(chain; Δx=Δx, L=L)
	
	    hm = heatmap!(
	        ax,
	        x_centers,
	        x_centers,
	        P,
	        colormap=ColorSchemes.cividis
	    )

		if !isnothing(environment)
			for i in 1:size(environment.grid, 1), j in 1:size(environment.grid, 2)
				if !environment.grid[i, j, 3]
					x = environment.grid[j, i, 1]
					y = environment.grid[j, i, 2]
		
					# Draw a black square centered at (x, y)
					rect = [
						Point2f(x - Δx/2, y - Δx/2),
						Point2f(x + Δx/2, y - Δx/2),
						Point2f(x + Δx/2, y + Δx/2),
						Point2f(x - Δx/2, y + Δx/2)
					]
					poly!(ax, rect, color=:black)
				end
			end
		end
	else
		scatter!(ax, chain[:, "x₀[1]"], chain[:, "x₀[2]"],marker=:+)
	end
	
	if show_source
		scatter!(ax, x₀[1], x₀[2], color="red", label="source", marker=:xcross, markersize=src_size, strokewidth=2)
		if show_legend
			axislegend(ax, location=:tr)
		end
		if show_as_heatmap
			return hm
		end
	end

	if show_as_heatmap
		return hm
	end
	
end

# ╔═╡ 66d5216d-66a8-4e33-9f36-54a0d4ec4459
"""
Helper function for viz_robot_grid().

Visualizes the robot's search grid over the environment, including optional overlays of data collection paths, posterior samples, and the true source location.

This function renders a heatmap of the environment’s layout and optionally overlays valid sampling grid points, the robot’s movement path and measurements, posterior samples from MCMC chains, and the ground truth source location.

# arguments
* `environment::Environment` – The environment struct generated by `generate_robot_grid_matrix()`. It must contain a field `grid` of type `Array{Union{Bool, Int64}, 3}`, where the first two entries are the x and y coordinates of grid locations and the third entry is a Boolean indicating accessibility.

# keyword arguments
* `data_collection::Union{DataFrame, Nothing}=nothing` – A DataFrame containing the robot’s path and measured radiation values; used to visualize the robot’s trajectory.
* `chain_data::Union{Nothing, DataFrame}=nothing` – Optional posterior chain data used to visualize source belief distributions (via `viz_chain_data!`).
* `fig_size::Int=800` – Controls the pixel resolution of the output figure.
* `show_grid::Bool=true` – If `true`, plots the robot’s valid sampling grid locations.
* `x₀::Union{Vector{Float64}, Nothing}=nothing` – If provided, marks the true source location as a red × marker.
* `scale_max::Float64=1e5` - the max source strength for the color bar.

# returns
* `Figure` – A `CairoMakie.Figure` object visualizing the environment, optionally overlaid with robot sampling grid, collected data, posterior beliefs, and source location.
"""
function viz_robot_grid!(
	ax,
	environment::Environment; 
	data_collection::Union{DataFrame, Nothing}=nothing,
	chain_data::Union{Nothing, DataFrame}=nothing,
	show_grid::Bool=true,
	x₀::Union{Vector{Float64}, Nothing}=nothing,
	scale_max::Real=1e2,
	view_chain_as_hm::Bool=false
)
    heatmap!(ax, environment.masked_env; colormap=reverse(ColorSchemes.grays))

    n_valid = count(environment.grid[:, :, 3] .== true)

    xs = zeros(Float64, n_valid)
    ys = zeros(Float64, n_valid)

    idx = 1
	if show_grid
		#loop through the grid and add true values to the scatter plot
	    for i in 1:size(environment.grid, 1), j in 1:size(environment.grid, 2)
	        if environment.grid[j, i, 3] == true
	            xs[idx] = environment.grid[i, j, 1]
	            ys[idx] = environment.grid[i, j, 2]
	            idx += 1
	        end
	    end
		scatter!(ax, xs, ys; color = :cyan, markersize = 10, label="search grid sampling point")
		if isnothing(x₀)
			axislegend(ax, position=:lb)
		end
	end

	if ! isnothing(chain_data)
		hm = viz_chain_data!(
			ax, 
			chain_data, 
			show_source=false, 
			show_as_heatmap=view_chain_as_hm,
			environment=environment
		)
	end

	if !isnothing(data_collection)	
		sc = viz_path!(ax, data_collection, scale_max=1e2)
	end

	if !isnothing(x₀)
		scatter!(ax, [x₀[1]], [x₀[2]], color="red", marker=:xcross, markersize=15, label="source", strokewidth=1)
		axislegend(ax, position=:lb)
	end

	visuals = Dict{Symbol, Any}()
	if !isnothing(data_collection)
		visuals[:sc] = sc
	end
	if view_chain_as_hm && !isnothing(chain_data)
		visuals[:hm] = hm
	end

	return visuals
end


# ╔═╡ 45014b50-c04b-4f42-83c3-775ec6cd6e3f
"""
Visualizes the robot's search grid over the environment, including optional overlays of data collection paths, posterior samples, and the true source location.

This function renders a heatmap of the environment’s layout and optionally overlays valid sampling grid points, the robot’s movement path and measurements, posterior samples from MCMC chains, and the ground truth source location.

# arguments
* `environment::Environment` – The environment struct generated by `generate_robot_grid_matrix()`. It must contain a field `grid` of type `Array{Union{Bool, Int64}, 3}`, where the first two entries are the x and y coordinates of grid locations and the third entry is a Boolean indicating accessibility.

# keyword arguments
* `data_collection::Union{DataFrame, Nothing}=nothing` – A DataFrame containing the robot’s path and measured radiation values; used to visualize the robot’s trajectory.
* `chain_data::Union{Nothing, DataFrame}=nothing` – Optional posterior chain data used to visualize source belief distributions (via `viz_chain_data!`).
* `fig_size::Int=800` – Controls the pixel resolution of the output figure.
* `show_grid::Bool=true` – If `true`, plots the robot’s valid sampling grid locations.
* `x₀::Union{Vector{Float64}, Nothing}=nothing` – If provided, marks the true source location as a red × marker.
* `scale_max::Float64=1e5` - the max source strength for the color bar.

# returns
* `Figure` – A `CairoMakie.Figure` object visualizing the environment, optionally overlaid with robot sampling grid, collected data, posterior beliefs, and source location.
"""
function viz_robot_grid(
	environment::Environment; 
	data_collection::Union{DataFrame, Nothing}=nothing,
	chain_data::Union{Nothing, DataFrame}=nothing,
	fig_size::Int=800,
	show_grid::Bool=true,
	x₀::Union{Vector{Float64}, Nothing}=nothing,
	scale_max::Real=1e2,
	view_chain_as_hm::Bool=false
)
    fig = Figure(size=(fig_size, fig_size))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="rad source search space")


	visuals = viz_robot_grid!(
		ax,
		environment,
		data_collection=data_collection,
		chain_data=chain_data,
		show_grid=show_grid,
		x₀=x₀,
		scale_max=scale_max,
		view_chain_as_hm=view_chain_as_hm
	)


	if !isnothing(data_collection)
		colorbar_tick_values = [10.0^e for e in range(0, log10(scale_max), length=6)]
		colorbar_tick_values[1] = 0.0
		colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]

		Colorbar(
			fig[1, 2],
			visuals[:sc],
			label = "counts", 
			ticks = (colorbar_tick_values, colorbar_tick_labels), 
			ticklabelsize=25, 
			labelsize=35)
	end
	
	if view_chain_as_hm
		Colorbar(
			fig[2, 1], 
		  	visuals[:hm],
			label="posterior density",
			ticklabelsize=20, 
			labelsize=30,
			width=fig_size * 0.75,  # scale for layout
			vertical=false)
	end
    return fig
end


# ╔═╡ d47b2021-7129-4a31-8585-2c7257489b1a
viz_robot_grid(grid_env)

# ╔═╡ deae0547-2d42-4fbc-b3a9-2757fcfecbaa
"""
Visualizes the robot path, measurement locations, and optionally the radiation field, source location, obstructions, and MCMC posterior data on a 2D plot.

This function overlays data collected during a simulation on top of a spatial map. If provided, it also plots the Poisson-predicted radiation field and source location from the analytical model, as well as any physical obstructions and posterior sample information from MCMC inference.

# arguments
* `path_data::DataFrame` – DataFrame containing the robot's path history and collected measurements. Must contain a column `"x [m]"` with position vectors.

# keyword arguments
* `x₀::Union{Nothing, Vector{Float64}}=nothing` – The true source location. If provided, it is plotted as a black cross.
* `rad_sim::Union{Nothing, RadSim}=nothing` – The analytical radiation model used to compute expected counts for visualization.
* `fig_size::Float64=1000.0` – Resolution (in pixels) for the figure size.
* `L::Float64=1000.0` – Length of the domain in meters (used if `rad_sim` is not provided).
* `scale_max::Float64=1e6` – Maximum count rate value for color scaling (used only if `rad_sim` is not provided).
* `z_slice::Int64=1` – Slice of the γ-matrix to visualize (default is the first slice).
* `save_num::Int64=0` – If greater than zero, saves the figure as `"save_num.png"`.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` – Optional list of obstructions (e.g., `Rectangle` or `Circle`) to be drawn on the plot.
* `chain_data::Union{Nothing, DataFrame}=nothing` – Optional MCMC posterior samples (e.g., from Turing.jl). If provided, they will be plotted on the figure.

# returns
* A `CairoMakie.Figure` object containing the assembled visualization.
"""
function viz_data_collection(
	path_data::DataFrame; 
	x₀::Union{Nothing, Vector{Float64}}=nothing, 
	rad_sim::Union{Nothing, RadSim}=nothing, 
	fig_size::Float64=1000.0, 
	L::Float64=1000.0, 
	scale_max::Float64=1e6, 
	z_slice::Int64=1, 
	save_num::Int64=0, 	
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing, chain_data::Union{Nothing, DataFrame}=nothing
)	    
	
	fig = Figure(size=(fig_size, fig_size))
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
		
		
		hm = viz_model_data!(ax, rad_sim, obstructions=obstructions)

		colorbar_tick_values = [10.0^e for e in range(0, log10(scale_max), length=6)]
		colorbar_tick_values[1] = 0.0

		colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]

		Colorbar(fig[1, 2], hm, label = "counts", ticks = (colorbar_tick_values, colorbar_tick_labels), ticklabelsize=25, labelsize=35)
	end

	sc = viz_path!(ax, path_data, scale_max=scale_max)

	if ! isnothing(x₀) || ! isnothing(rad_sim)
		scatter!([source[1]], [source[2]], color="black", marker=:xcross, markersize=25, label="source", strokewidth=1)

		#axislegend(location=:tr)
	end

	if ! isnothing(chain_data)
		viz_chain_data!(ax, chain_data, show_source=false)
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

# ╔═╡ 2fe974fb-9e0b-4c5c-9a5a-a5c0ce0af065
function viz_chain_data(
	chain::DataFrame; 
	res::Float64=800.0, 
	L::Float64=L, 
	show_source::Bool=true, 
	path_data::Union{Nothing, DataFrame}=nothing, 
	scale_max::Real=200.0,
	show_as_heatmap::Bool=false
)
	
	fig = Figure(size = (res, res))
	ax = Axis(fig[1, 1], aspect=DataAspect())
	viz_chain_data!(
		ax, 
		chain, 
		show_source=show_source, 
		show_as_heatmap=show_as_heatmap
	)

	if !isnothing(path_data)
		sc = viz_path!(ax, path_data, scale_max=scale_max)
		colorbar_tick_values = [10.0^e for e in range(0, log10(scale_max), length=6)]
		colorbar_tick_values[1] = 0.0

		colorbar_tick_labels = [@sprintf("%.0e", val) for val in colorbar_tick_values]

		Colorbar(fig[1, 2], sc, label = "counts", ticks = (colorbar_tick_values, colorbar_tick_labels), ticklabelsize=15, labelsize=25)
	end

	xlims!(-1, L+1)
	ylims!(-1, L+1)
	
	return fig
end

# ╔═╡ aa72cf61-839d-4707-95c8-0a9230e77d56
md"## `Visualize` - Posterior"

# ╔═╡ f4d234f9-70af-4a89-9a57-cbc524ec52b4
function viz_posterior(chain::DataFrame; Δx::Float64=Δx, L::Float64=L)
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
	#=hb = hexbin!(
		ax_b, chain[:, "x₀[1]"], chain[:, "x₀[2]"], colormap=colormap, bins=round(Int, L/Δx)
	)=#
	hm = viz_chain_data!(
		ax_b, 
		chain,
		show_source=true,
		show_as_heatmap=true,
		Δx=Δx, 
		L=L,
		show_legend=false,
		src_size=7.0
	)
	Colorbar(fig[2, 2], hm, label="density")

	# show ground-truth
	vlines!(ax_t, I, color="red", linestyle=:dash)
	scatter!(ax_b, [x₀[1]], [x₀[2]], marker=:+, color="red")
	
	return fig
end

# ╔═╡ 95837cad-192d-46b4-aaa4-c86e9b1d1c09
md"# Exploration control"

# ╔═╡ 8c18d4e8-fd5c-4fd4-8f1e-516615a9e5f0
md"## Entropy"

# ╔═╡ ed8381e3-7a83-4afc-a95d-5cbd03b7e852
"""
entropy of the posterior over source location.
"""
function entropy(chain::DataFrame)
	P = chain_to_P(chain)
	entropy = sum(
		[-P[i] * log2(P[i]) for i in eachindex(P) if P[i] > 0.0]
	)
	return entropy
end

# ╔═╡ eafb66bc-6da3-4570-b62a-922627e6ccde
md"## `Visualize` - Simulation chain entropy"

# ╔═╡ 600a5f36-cfa2-4848-8984-44f0ae54ed67
function viz_sim_chain_entropy(chains::Dict)

	num_sims = length(chains)
	entropys = [entropy(chains[i]) for i=1:num_sims]

	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="step", ylabel="posterior entropy")

	lines!(ax, 1:num_sims, entropys)

	fig
end

# ╔═╡ 28f9219a-6758-4aa3-8f96-f5b2e8a8e5d7
function viz_sim_chain_σ(chains::Dict; window_size::Int=10, use_diff::Bool=true)
	function movmean(v::Vector{<:Real}, window::Int)
	    n = length(v)
	    result = similar(v, Float64)
	    for i in 1:n
	        start_idx = max(1, i - window + 1)
	        result[i] = mean(@view v[start_idx:i])
	    end
	    return result
	end
	
	num_sims = length(chains)
	σs = zeros(num_sims)

	#Calculate the Euclidean norm or L2 norm or root sum of squares RSS.
	for i=1:num_sims
		σₓ₁ = std(chains[i][:, "x₀[1]"])
		σₓ₂ = std(chains[i][:, "x₀[2]"])
		σ_total = sqrt(σₓ₁^2 + σₓ₂^2)
		
		σs[i] = σ_total
	end

	σs_smooth = movmean(σs, window_size)



	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="step", ylabel="posterior L2 norm")

	if use_diff
		σs_diff = diff(σs_smooth)
		lines!(ax, 1:length(σs_diff), movmean(σs_diff, window_size))
	else
		lines!(ax, 1:num_sims, σs_smooth)		
	end

	fig
end

# ╔═╡ bb94ce77-d48c-4f6d-b282-96197d6e7b6b
md"# Thompson sampling"

# ╔═╡ f55544f3-413d-44c5-8e81-37a5f017b460
md"## Thompson sampling for simulated space"

# ╔═╡ 161833bd-aa39-4bab-98b6-be10b6d3653f


# ╔═╡ 76bc6fcb-0018-40dd-9709-65bf9d223615
md"### building overlap functions"

# ╔═╡ 12729385-04ea-4c8b-ab08-e114b4f4172d
"""
Checks whether a 2D position lies within a rectangular obstruction.

This function determines whether the point `pos` is inside the axis-aligned rectangle specified by its center and dimensions.

# rectangle arguments
* `pos::Vector{Float64}` – A 2D position `[x, y]` to check for overlap.
* `rect::Rectangle` – The rectangular obstruction defined by its center, width, and height.

# returns
* `Bool` – `true` if `pos` lies within the rectangle, `false` otherwise.
"""
function overlaps(pos::Vector{Float64}, rect::Rectangle)
	x, y = pos
	cx, cy = rect.center
	hw, hh = rect.width / 2, rect.height / 2
	return (cx - hw ≤ x ≤ cx + hw) && (cy - hh ≤ y ≤ cy + hh)
end

# ╔═╡ 32228db5-bf76-4633-ab5e-224f95459cc9
"""
Checks whether a 2D position lies within a circular obstruction.

This function determines whether the point `pos` is within or on the boundary of the circle defined by its center and radius.

# circle arguments
* `pos::Vector{Float64}` – A 2D position `[x, y]` to check for overlap.
* `circ::Circle` – The circular obstruction defined by its center and radius.

# returns
* `Bool` – `true` if `pos` lies within or on the boundary of the circle, `false` otherwise.
"""
function overlaps(pos::Vector{Float64}, circ::Circle)
	x, y = pos
	cx, cy = circ.center
	return (x - cx)^2 + (y - cy)^2 ≤ circ.radius^2
end

# ╔═╡ ddc23919-17a7-4c78-86f0-226e4d447dbe
"""
Moves the robot `n` times in a specified direction by modifying the robot path in-place.

This function either performs `n` individual steps or one aggregated movement of `n × Δx`, depending on the `one_step` flag. Movement halts early if it would result in collision with an obstruction or leave the defined environment bounds.

# arguments
* `robot_path::Vector{Vector{Float64}}` – The robot's current path, with the last element representing its current position.
* `direction::Symbol` – The direction to move. Must be one of `:up`, `:down`, `:left`, or `:right`.
* `n::Int` – The number of steps to move.

# keyword arguments
* `Δx::Float64=Δx` – The grid spacing value.
* `one_step::Bool=false` – If `true`, moves the robot `n` steps in one large movement. If `false`, moves one step at a time, allowing for intermediate checks or measurements.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` – Optional list of obstructions; movement will halt if any are encountered.
* `L::Float64=1000.0` – The length of the square environment; movement outside this boundary is not allowed.

# modifies
* `robot_path` – Updated in-place with the new robot position(s), depending on movement conditions and the `one_step` setting.
"""
function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol, n::Int; Δx::Float64=Δx, one_step::Bool=false, obstructions::Union{Nothing, Vector{Obstruction}}=nothing, L::Float64=1000.0)
		if one_step
			pos = copy(robot_path[end])
   			Δ = get_Δ(direction; Δx=Δx)

			for _ in 1:n
        		candidate = pos .+ Δ
        		if obstructions !== nothing && any(
						overlaps(
							candidate, obs
						) for obs in obstructions
					)
           	 		break
				elseif any(x -> x <= 0.0 || x >= L, candidate)
					break
        		end
				#need to add logic  to make sure we're not moving off map i.e. candidate can't be greater than L or less than 0
        		pos .= candidate
    		end
			push!(robot_path, pos)
		else
			for i = 1:n
				move!(robot_path, direction, Δx=Δx)
			end
		end
	end

# ╔═╡ 50e623c0-49f6-4bb5-9b15-c0632c3a88fd
begin
	#Δx = 10.0 # m (step length for robot)
	robot_path = [[0.0, 0.0]] # begin at origin
	
	move!(robot_path, :up, 5)
	move!(robot_path, :right, 7)
	move!(robot_path, :up, 10)
	move!(robot_path, :right, 15)
	
	data = DataFrame(
		"time" => 1:length(robot_path),
		"x [m]" => robot_path,
		"counts" => [sample_model(x, test_rad_sim, I=I) for x in robot_path]
	)
end

# ╔═╡ dac856e6-f651-49d0-994e-46c8296a3d30
data

# ╔═╡ 9fbe820c-7066-40b5-9617-44ae0913928e
viz_data_collection(data, rad_sim=test_rad_sim)

# ╔═╡ 13ff8f6a-7bb2-41a0-83ac-7c9fca962605
chain = DataFrame(
	sample(rad_model(data), NUTS(), MCMCThreads(), 500, 2)
)

# ╔═╡ ea2dc60f-0ec1-4371-97f5-bf1e90888bcb
 viz_chain_data(chain, show_as_heatmap=true)

# ╔═╡ 4bb02313-f48b-463e-a5b6-5b40fba57e81
viz_posterior(chain)

# ╔═╡ 8b98d613-bf62-4b2e-9bda-14bbf0de6e99
"""
Given the robot path, returns a tuple of optional directions the robot could travel in next.

# arguments
* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `L::Float64` - the width/length of the space being explored.
# keyword arguments
* `Δx::Float64=10.0` - step size of the robot.
* `allow_overlap::Bool=false` - if set to true, allows the robot to backtrack over the previously visited position.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - vector of obstruction objects, currently only accepting Rectangle and Circle types.
# returns
* `Vector{Symbol}` – A list of direction symbols (`:up`, `:down`, `:left`, `:right`) representing valid next moves for the robot based on the current position, grid boundaries, accessibility, and overlap settings. The directions are filtered to avoid backtracking unless `allow_overlap=true`.
"""
function get_next_steps(
	robot_path::Vector{Vector{Float64}}, 
	L::Float64; 
	Δx::Float64=10.0,
	allow_overlap::Bool=false,
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing
)

	current_pos = robot_path[end]

	directions = Dict(
        :up    => [0.0, Δx],
        :down  => [0.0, -Δx],
        :left  => [-Δx, 0.0],
        :right => [Δx, 0.0]
    )

	last_step = length(robot_path)>1 ? robot_path[end] .- robot_path[end-1] : nothing
	

	function is_valid(dir)
		function unit(v)
		    n = norm(v)
		    n ≈ 0.0 ? v : v ./ n
		end
	    step = directions[dir]
	    new_pos = current_pos .+ step
	    in_bounds = all(0.0 .≤ new_pos .≤ L)
	    dir_unit = unit(step)
	    last_unit = isnothing(last_step) ? dir_unit : unit(last_step)
	    not_backtrack = allow_overlap || isnothing(last_step) || dot(dir_unit, last_unit) > -0.9
	    not_blocked = isnothing(obstructions) || all(obs -> !overlaps(new_pos, obs), obstructions)
	    return in_bounds && not_backtrack && not_blocked
	end

	return Tuple(filter(is_valid, keys(directions)))

end

# ╔═╡ ff90c961-70df-478a-9537-5b48a3ccbd5a
md"## simulate source localization"

# ╔═╡ 44d81172-2aef-4ef1-90e9-6a169e92f9ff
md"## `Example Sim`"

# ╔═╡ ae92f6ae-298d-446d-b379-ee2190ef1915
start = [65, 70]

# ╔═╡ ad54d1fa-b3e7-4aeb-96f4-b5d15dee38d5
md"### simulation control"

# ╔═╡ 5c03dc8e-6484-4a73-8cb7-eb43aa382a9d
begin
	# change this to the number of steps you want the robot to take before giving up
	# without obstructions
	num_steps_sim = 90
	#with obstructions
	num_steps_sim_obst = 40

	#num of MCMC chains & samples
	num_mcmc_chain = 4
	num_mcm_sample = 150
	

end

# ╔═╡ e7023831-5c03-4f53-95f4-ab837bced1b2
#print(simulation_data)

# ╔═╡ 22a012c1-4169-4959-af47-9d4b01691ae9
#test_rad_sim_obstructed

# ╔═╡ 2f7a3d49-1864-4113-b173-ee7e8c9e62a4
md"## `Example Sim` - with obstructions"

# ╔═╡ ef7ff4ec-74ac-40b9-b68b-dbc508e50bef
simulation_data_obst, simulation_chains_obst = SimulationSpace.simulate(test_rad_sim_obstructed, num_steps_sim_obst, save_chains=true, num_mcmc_samples=num_mcm_sample, num_mcmc_chains=num_mcmc_chain, robot_start=start, obstructions=obstructions, exploring_start=true, spiral=false, num_exploring_start_steps=1)

# ╔═╡ 9d0795fa-703e-47a4-8f1e-fe38b9d604b4
simulation_chains_obst

# ╔═╡ b7673342-70f2-4cbb-869e-1b67f9ee7235
viz_sim_chain_entropy(simulation_chains_obst)

# ╔═╡ 97de1cb8-9c72-440b-896a-a1f1d24e46f5
viz_sim_chain_σ(simulation_chains_obst)

# ╔═╡ 3ff17eaf-974d-4bf0-b75f-d3ef473730bf
begin
	for i=1:length(simulation_chains_obst)
		#viz_data_collection(simulation_data_obst[1:i, :], rad_sim=test_rad_sim_obstructed, obstructions=obstructions, chain_data=simulation_chains_obst[i], save_num=i)
	end
end

# ╔═╡ ea505dc1-a18f-408f-bff8-3b488c49fdb0
@bind chain_val_obst PlutoUI.Slider(1:size(simulation_data_obst, 1)-1, show_value=true)

# ╔═╡ ac2dd9e7-0547-4cda-acf5-845d12d87626
viz_data_collection(simulation_data_obst[1:chain_val_obst, :], rad_sim=test_rad_sim_obstructed, obstructions=obstructions, chain_data=simulation_chains_obst[chain_val_obst])

# ╔═╡ d14fc2b4-ad11-4506-a580-06bfefede40b
 viz_posterior(simulation_chains_obst[chain_val_obst])

# ╔═╡ a53b3039-eb9e-45aa-914f-034d2a5b6e01
md"# Save sim data"

# ╔═╡ 40cfe92d-b707-4b22-b3f9-228e5a0df7b2
md"# Batch Test"

# ╔═╡ 0c2d090c-82c8-466d-aea7-140b4422d254
md"## latin hypercube sample starts"

# ╔═╡ 5e5c4e18-63b9-4b2b-bf75-52c77ec3d0fe
"""
Using latin hypercube sampling, generate `num_samples` of pseudo uniformly distributed sample start locations for the robot.

# keyword arguments
* `num_samples::Int=15` - number of sample start locations.
* `L::Float64=L` - space size.
* `Δx::Float64=Δx` - discretization.
# returns
* `Vector{Vector{Int}}` – A list of robot starting positions expressed as integer grid indices `[i, j]`. The sample points are generated using Latin Hypercube Sampling (LHS) to ensure pseudo-uniform coverage of the space. If obstructions are provided, all returned points are guaranteed not to overlap with any obstruction region.
"""
function gen_sample_starts(
	;num_samples::Int=15, 
	L::Float64=L, 
	Δx::Float64=Δx, 
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing
)

	@assert num_samples < 100 "please limit num_samples to less than 100"
	#get latin hypercube samples
	lhc_samples = LHCoptim(num_samples, 2, 10)[1] .* L /num_samples

	#convert to vectors of grid indicies
	r_starts = [[floor(Int, lhc_samples[i, 1] / Δx), 
                 floor(Int, lhc_samples[i, 2] / Δx)]
                 for i in 1:size(lhc_samples, 1)]

	#if obstructions are provided, check to make sure no overlap occurs
	if !isnothing(obstructions)
		for coords in r_starts
			x₁ = ((coords[1] - 1) * Δx) + 0.5 * Δx
	    	x₂ = ((coords[2] - 1) * Δx) + 0.5 * Δx
			#if overlap is found with obstruction, rerun LHC algorithm
			if !all(obs -> !overlaps([x₁,x₂] , obs), obstructions)
				return gen_sample_starts(num_samples=num_samples, L=L, Δx=Δx, obstructions=obstructions)
			end
		end	
	end

	return r_starts
end

# ╔═╡ fd3393e0-9e08-41e6-a6d2-c28743cb1a68
robot_starts = gen_sample_starts(num_samples=12, obstructions=obstructions)

# ╔═╡ e75b8aae-8da8-45e8-8405-103f77a3cca6
md"## run batch test"

# ╔═╡ c84fc2c3-5d44-49dd-a176-cf7277b4ef30
robot_starts

# ╔═╡ e5ead52b-c407-400d-9a26-fca9b61556f3
begin

	#=some_test_start = [[85, 85]]=#
	#=
	batch_test = run_batch(
	test_rad_sim_obstructed, 
	robot_starts, 
	num_mcmc_samples=150,
	num_mcmc_chains=4,
	I=I,
	L=L,
	Δx=Δx,
	allow_overlap=false,
	x₀=[250.0, 250.0],
	z_index=1,
	obstructions=obstructions,
	exploring_start=true,
	num_exploring_start_steps=15,
	spiral=true,
	r_check=70.0,
	r_check_count=10,
	meas_time=1.0,
	num_replicates=10,
	filename="test_batch")=#
end

# ╔═╡ 5e5bf646-0a05-4405-8563-86abe65d6fca
#=
test_params(
	test_rad_sim, 
	robot_starts,
	filename="no_obstructions")
=#

# ╔═╡ 75cda12e-3a12-44b4-aa51-ef60588fee49
md"# Experimental space"

# ╔═╡ de432aa4-b320-4598-aedd-32ca2b74be52
"""
Given the robot path, returns a tuple of optional directions the robot could travel in next.

# arguments
* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `environment::Environment` - environment struct generated by the `generate_robot_grid_matrix()` function, which should contain a field named grid of type `Array{Union{Bool, Int64}, 3}` where the first two entries are the x and y coordinates of each grid location for the robot search space and the third entry is the matrix of boolean values representing accessibility of locations.
# keyword arguments
* `allow_overlap::Bool=false` - if set to true, allows the robot to backtrack over the previously visited position.
# returns
* `Vector{Symbol}` – A list of direction symbols (`:up`, `:down`, `:left`, `:right`) representing valid next moves for the robot based on the current position, grid boundaries, accessibility, and overlap settings. The directions are filtered to avoid backtracking unless `allow_overlap=true`.
"""
function get_next_steps(
	robot_path::Vector{Vector{Float64}}, 
	environment::Environment;
	allow_overlap::Bool=false
)

	#this is needed for src file where type assertions for structs get weird
	@assert hasfield(typeof(environment), :grid) "environment struct, $(environment), must contain the :grid field."

	#ensure equal grid spacing by ensuring equal entries by row for x's, column for y's
	for i=1:size(environment.grid[:, :, 1], 1)
		@assert environment.grid[i, :, 1] == environment.grid[1, :, 1] "row $i is not equal to row 1... all rows must have equal entries."
	end

	for j=1:size(environment.grid[:, :, 1], 2)
		@assert environment.grid[:, j, 2] == environment.grid[:, 1, 2] "column $j is not equal to column 1... all columns must have equal entries."
	end

	#get indicies that most closely match the current position
	xs 			= environment.grid[1, :, 1]
	ys 			= environment.grid[:, 1, 2]
	current_pos = robot_path[end]
	x_index 	= argmin(abs.(xs .- current_pos[1]))
	y_index 	= argmin(abs.(ys .- current_pos[2]))
	pos_index 	= [x_index, y_index]

	#establish directional index change
	directions 	= Dict(
        :up    => [0, 1],
        :down  => [0, -1],
        :left  => [-1, 0],
        :right => [1, 0]
    )

	prev_pos = length(robot_path)>1 ? robot_path[end-1] : robot_path[end]
	prev_index = [argmin(abs.(xs .- prev_pos[1])), argmin(abs.(ys .- prev_pos[2]))]

	#collect valid directions
	valid_directions = [
        dir for (dir, Δ) in directions 
        if (0 .< (x_index .+ Δ[1]) ≤ size(environment.grid, 1)) &&
           (0 .< (y_index .+ Δ[2]) ≤ size(environment.grid, 2)) &&
           environment.grid[x_index + Δ[1], y_index + Δ[2], 3] == true &&
           (allow_overlap || (x_index + Δ[1], y_index + Δ[2]) != prev_index)
    ]

	return valid_directions

end

# ╔═╡ ec9e4693-771c-467d-86cc-ab2ba90019fe
"""
`step_spiral!` helper function, advances the spiral controller by one step and updates the robot path accordingly.

This function moves the robot in a growing outward spiral pattern by stepping in the current direction of the spiral controller. After completing a leg of movement, it rotates the direction and increases the leg length every two turns. If the next step would enter an obstruction or exceed the environment boundary, a random valid direction is chosen instead.

# arguments
* `sc::SpiralController` – The spiral controller managing current position, direction, and step logic.
* `robot_path::Vector{Vector{Float64}}` – The history of the robot's positions; the current position will be appended after the step.

# keyword arguments
* `Δx::Float64=10.0` – Step size in meters.
* `L::Float64=1000.0` – Width/length of the square domain; used to enforce boundary constraints.
* `allow_overlap::Bool=false` – If `false`, previously visited locations are avoided unless movement is otherwise blocked.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` – Vector of static obstructions (`Rectangle`, `Circle`, etc.) that the robot must avoid.

# returns
* The updated position as a `Vector{Float64}` after the spiral step is executed.
"""
function step_spiral!(
    sc::SpiralController,
	robot_path::Vector{Vector{Float64}};
    Δx::Float64=10.0,
    L::Float64=1000.0,
    allow_overlap::Bool=false,
    obstructions::Union{Nothing, Vector{Obstruction}}=nothing
)
    #gt current direction and step delta
    dir = sc.directions[sc.dir_idx]
    Δ = get_Δ(dir; Δx=Δx)
 	new_pos = sc.pos .+ Δ

	#obstruction or boundary check
    if !isnothing(obstructions)
        blocked = any(obs -> overlaps(new_pos, obs), obstructions) || any(x -> x < 0.0 || x > L, new_pos)
        if blocked
            valid_dirs = get_next_steps([sc.pos], 
										L, 
										Δx=Δx, 
										allow_overlap=allow_overlap, obstructions=obstructions
									   )
            if isempty(valid_dirs)
				push!(robot_path, copy(sc.pos))
                return sc.pos  # stuck, return current position
            else
                new_dir = rand(valid_dirs)
                Δ = get_Δ(new_dir, Δx=Δx)
                new_pos = sc.pos .+ Δ
                sc.pos .= new_pos
				push!(robot_path, copy(sc.pos))
                return new_pos
            end
        end
    end
	

    #step
    sc.pos .= new_pos

    #trck how many steps taken this leg
    sc.leg_count += 1

    #aftr completing a full leg, updte dir
    if sc.leg_count == sc.step_size
        sc.leg_count = 0
        sc.dir_idx = mod1(sc.dir_idx + 1, 4)
        if iseven(sc.dir_idx)  #incr step size every 2 turns
            sc.step_size += sc.step_increment
        end
    end

	push!(robot_path, copy(sc.pos))
    return copy(sc.pos)
end

# ╔═╡ a2154322-23de-49a6-9ee7-2e8e33f8d10c
"""
Given the robot path, finds the best next direction the robot to travel using Thompson sampling of the posterior.

# arguments
* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
* `L::Float64` - the width/length of the space being explored.
# keyword arguments
* `Δx::Float64=2.0` - step size of the robot.
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - vector of obstruction objects, currently only accepting Rectangle and Circle types.
# returns
* `Symbol` – The best direction for the robot to move next, chosen from `:up`, `:down`, `:left`, or `:right`, based on the Thompson-sampled posterior estimate of the source location. Returns `:nothing` if no valid direction is available and `allow_overlap=true`. If `allow_overlap=false` and no direction is valid, the function retries recursively with `allow_overlap=true`.
"""
function thompson_sampling(
	robot_path::Vector{Vector{Float64}}, 
	chain::DataFrame;
	L::Float64=50.0,
	Δx::Float64=2.0,
	allow_overlap::Bool=false,
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing)

	#find direction options from left, right, up, down within domain
	direction_options = get_next_steps(robot_path, L, allow_overlap=allow_overlap, obstructions=obstructions)

	if length(direction_options) < 1 && allow_overlap == true
		@warn "found no viable direction options with overlap allowed, returning nothing"
		return :nothing
	end

	best_direction = :nothing
	greedy_dist = Inf

	#randomly sample from the chain
	rand_θ = chain[rand(1:nrow(chain)), :]
	loc = robot_path[end]

	#loop through direction options and pick the one leading closest to sample
	for direction in direction_options
		new_loc = loc .+ get_Δ(direction)
		dist = norm([rand_θ["x₀[1]"]-new_loc[1], rand_θ["x₀[2]"]-new_loc[2]])
		if dist < greedy_dist
			greedy_dist = dist
			best_direction = direction
		end
	end

	#if we can't find any direction at allow and overlap isn't allowed, redo w/ overlap
	if best_direction == :nothing && allow_overlap == false
		@warn "best direction == nothing, switching to allow overlap"
		return thompson_sampling(
			robot_path, 
			chain,
			L=L,
			Δx=Δx,
			allow_overlap=true,
			obstructions=obstructions)
	end
	
	return best_direction

end

# ╔═╡ 1c0fee71-29a2-4fa1-93db-e1213ed88bb0
md"## Thompson sampling for experimental space"

# ╔═╡ 54e52416-6c81-4dae-be10-2ddd1449dbfa
"""
Given the robot path, finds the best next direction the robot to travel using thompson sampling of the posterior.

# arguments
* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `environment::Environment` - environment struct generated by the `generate_robot_grid_matrix()` function, which should contain a field named grid of type `Array{Union{Bool, Int64}, 3}` where the first two entries are the x and y coordinates of each grid location for the robot search space and the third entry is the matrix of boolean values representing accessibility of locations.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
# keyword arguments
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
# returns
* `Symbol` – The direction the robot should move next, selected from `:up`, `:down`, `:left`, or `:right` based on the sampled posterior estimate of the source location. If no valid direction is found, returns `:nothing`, or retries with `allow_overlap=true` if not already enabled.
"""
function thompson_sampling(
	robot_path::Vector{Vector{Float64}}, 
	environment::Environment,
	chain::DataFrame;
	allow_overlap::Bool=false
)

	#find direction options from left, right, up, down within domain
	direction_options = get_next_steps(
		robot_path, 
		environment,
		allow_overlap=allow_overlap
	)

	if length(direction_options) < 1 && allow_overlap == true
		@warn "found no viable direction options with overlap allowed, returning nothing"
		return :nothing
	end

	best_direction = :nothing
	greedy_dist = Inf

	#randomly sample from the chain
	rand_θ = chain[rand(1:nrow(chain)), :]
	target = [rand_θ["x₀[1]"], rand_θ["x₀[2]"]]
	loc = robot_path[end]

    # Loop through each valid direction and calculate distance
    for direction in direction_options
        Δ = get_Δ(direction, Δx=environment.Δ)
        new_location = loc .+ Δ  # Apply grid spacing

        # Compute Euclidean distance to the sampled location
        dist = norm(new_location .- target)

        if dist < greedy_dist
            greedy_dist = dist
            best_direction = direction
        end
    end

	#if we can't find any direction at allow and overlap isn't allowed, redo w/ overlap
	if best_direction == :nothing && allow_overlap == false
		@warn "best direction == nothing, switching to allow overlap"
		return thompson_sampling(
			robot_path, 
			environment,
			chain,
			allow_overlap=true
		)
	end
	
	return best_direction

end

# ╔═╡ 52296a3f-9fad-46a8-9894-c84eb5cc86d7
"""
Runs a simulation by placing a robot, calculating a posterior, sampling the posterior using Thompson sampling, then making a single step and repeating `num_steps` times.

# arguments
* `rad_sim::RadSim` - the radiation simulation RadSim to sample from.
* `num_steps::Int64` - set the max number of steps to simulate movement.
# keyword arguments
* `robot_start::Vector{Int64}=[0, 0]` - the grid indicies for the robot to start the simulation.
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `I::Float64=I` - source strength.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=2.0` - step size of the robot.
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
* `x₀::Vector{Float64}=[250.0, 250.0]` - source location, this tells the simulation to stop if the current location is within Δx of x₀.
* `save_chains::Bool=false` - set to true to save the MCMC simulation chain data for every step.
* `z_index::Int=1` - sets the current z index of the γ_matrx, for now keep at 1.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - vector of obstruction objects, currently only accepting Rectangle and Circle types.
* `exploring_start::Bool=true` - if set to false, will begin by taking single steps only towards the greedy choice from Thompson sampling, otherwise will take decreasingly large steps.
* `num_exploring_start_steps::Int=10` - controls the size and number of exploring start steps. For example, a value of 10 will make the robot take 10 steps initially, followed by 9, then 8 etc...
* `r_check::Float64=70.0` - the radius to check for `r_check_count` number of samples to determine if a large next step should be taken.
* `r_check_count::Int=10` - the number of samples within radius `r_check` to determine if a large next step should be taken.
* `meas_time::Float64=1.0` - set the temporal cost of collecting data.
* `disable_log::Bool=true` - set to false to allow logging by Turing.jl.
# returns
* `sim_data::DataFrame` – A DataFrame containing the simulation results. Each row corresponds to a step in the robot's path and includes the following columns:
  - `"time"` – Cumulative time at each step (including travel and measurement).
  - `"x [m]"` – 2D position of the robot in meters.
  - `"counts"` – Measured radiation counts at each location.

* `sim_chains::Dict{Int, DataFrame}` *(only if `save_chains=true`)* – A dictionary mapping each simulation step index to the corresponding MCMC chain output as a DataFrame. Each chain represents the posterior samples for source location and intensity at that step.
"""
function simulate(
	rad_sim::RadSim,
	num_steps::Int64; 
	robot_start::Vector{Int64}=[0, 0], 
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	I::Float64=I,
	L::Float64=L,
	Δx::Float64=Δx,
	allow_overlap::Bool=false,
	x₀::Vector{Float64}=[250.0, 250.0],
	save_chains::Bool=false,
	z_index::Int=1,
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing,
	exploring_start::Bool=true,
	num_exploring_start_steps::Int=15,
	spiral::Bool=true,
	r_check::Float64=70.0,
	r_check_count::Int=10,
	meas_time::Float64=1.0,
	disable_log::Bool=true
)
	# set up initial position and take measurement
	x_start = [(robot_start[i]-1) * Δx for i=1:2]
	c_start = sample_model(x_start, rad_sim, I=I, Δx=Δx, z_index=z_index)

	# data storage
	sim_chains = Dict()
	sim_data = DataFrame(
		"time" => [meas_time],
		"x [m]" => [x_start],
		"counts" => [c_start]
	)
	robot_path = [x_start]

	# exploration parameters
	if spiral
		spiral_control = init_spiral(copy(robot_path[end]), step_init=1, step_incr=2)
	end
	if exploring_start
		expl_start_steps = [num_exploring_start_steps - i >= 1 ? num_exploring_start_steps - i : 1 for i in 0:num_steps-1]
	end
	

	######################################################
	# simulation loop #
	for iter = 1:num_steps
		model = rad_model(sim_data)
		if disable_log
			Logging.with_logger(NullLogger()) do
			model_chain = DataFrame(
				sample(model, NUTS(), MCMCThreads(), num_mcmc_samples, num_mcmc_chains, progress=false, thin=5)
			)
			end
		else
			model_chain = DataFrame(
				sample(model, NUTS(), MCMCThreads(), num_mcmc_samples, num_mcmc_chains, progress=false, thin=5)
			)
		end
		#NUTS(100, 0.65, max_depth=7)

		if save_chains
			sim_chains[iter] = model_chain
		end

		#if the source is found, break here
		if norm([robot_path[end][i] - x₀[i] for i=1:2]) <= (2 * Δx^2)^(0.5)
			@info "Source found at step $(iter), robot at location $(robot_path[end])"
			break
		end

		#spiral at the beginning
		if spiral && (iter <= num_exploring_start_steps)

			spiral_step_spacing = 4 * Δx
			new_pos = step_spiral!(spiral_control, 
								   robot_path,
								   Δx=spiral_step_spacing, 
								   L=L, 
								   obstructions=obstructions
								  )
			c_measurement = sample_model(robot_path[end], rad_sim, I=I, Δx=Δx, z_index=z_index)
			Δt_travel = norm(robot_path[end] .- robot_path[end-1]) / r_velocity
			push!(
				sim_data,
				Dict("time" => sim_data[end, "time"] + Δt_travel + meas_time, 
				"x [m]" => robot_path[end], 
				"counts" => c_measurement
				)
			)
		#not spiraling
		else
			# use Thompson sampling to find the best direction
			best_direction = thompson_sampling(
				robot_path, 
				model_chain,
				L=L,
				Δx=Δx,
				allow_overlap=allow_overlap,
				obstructions=obstructions
			)

			#debug code, returns what we have so far in the case of an issue.
			#best_direction shouldn't ever be nothing.
			if best_direction == :nothing
				@warn "iteration $(iter) found best_direction to be :nothing"
				if save_chains
					return sim_data, sim_chains
				else
					return sim_data
				end
			end
	
			# check how many samples takin within r_check radius
			num_within_r_check = sum(
	    		norm(pos .- robot_path[end]) ≤ r_check for pos in robot_path
			)
			# if criteria met, move a lot at once
			if num_within_r_check >= r_check_count
				#move less if counts are high recently
				move_dist = sim_data[iter-1, "counts"] < 2   ? 15 :
				            sim_data[iter-1, "counts"] < 5   ? 10 :
				            sim_data[iter-1, "counts"] < 10  ? 5  :
				            sim_data[iter-1, "counts"] < 30  ? 2  : 1
			else
				move_dist = 1
			end

			#if exploring start, explore first, then adjust movement afterwards
			if exploring_start
			    proposed_dist = expl_start_steps[iter]
			    move_dist = proposed_dist > 1 ? proposed_dist : move_dist
			end
			
			#Move the robot
			move!(
				robot_path, 
				best_direction, 
				move_dist,
      			Δx=Δx, 
				one_step=true, 
				L=L, 
				obstructions=obstructions
			)
			#collect data
			c_measurement = sample_model(robot_path[end], rad_sim, I=I, Δx=Δx, z_index=z_index)
			Δt_travel = norm(robot_path[end] .- robot_path[end-1]) / r_velocity
			push!(
				sim_data,
				Dict("time" => sim_data[end, "time"] + Δt_travel + meas_time, 
				"x [m]" => robot_path[end], 
				"counts" => c_measurement
				)
			)
		end
	end
	# end simulation loop #
	######################################################

	return save_chains ? (sim_data, sim_chains) : sim_data
end

# ╔═╡ f847ac3c-6b3a-44d3-a774-4f4f2c9a195d
simulation_data, simulation_chains = simulate(test_rad_sim, num_steps_sim, save_chains=true, num_mcmc_samples=num_mcm_sample, num_mcmc_chains=num_mcmc_chain, robot_start=start,  exploring_start=true, num_exploring_start_steps=45, spiral=true)

# ╔═╡ c6b9ca97-7e83-4703-abb9-3fd43daeb9a7
viz_sim_chain_entropy(simulation_chains)

# ╔═╡ f063123b-bab8-435c-b128-0dc72d31b5fb
viz_sim_chain_σ(simulation_chains)

# ╔═╡ f5ea3486-4930-42c2-af1b-d4a17053976a
@bind chain_val PlutoUI.Slider(1:size(simulation_data, 1)-1, show_value=true)

# ╔═╡ 9a1fa610-054b-4b05-a32b-610f72329166
viz_data_collection(DataFrame(simulation_data[1:chain_val, :]), chain_data=simulation_chains[chain_val],  rad_sim=test_rad_sim)

# ╔═╡ 4a0c8aab-2424-441d-a8c7-9f8076ecbae7
 viz_posterior(simulation_chains[chain_val])

# ╔═╡ 34527801-4098-4ffe-99c0-5abbdd99ee55
begin
	sim_data = Dict(
		"obstr" => Dict(
			"sim_data" => simulation_data_obst,
			"chains" => simulation_chains_obst
		),
		"no_obstr" => Dict(
			"sim_data" => simulation_data,
			"chains" => simulation_chains
		)
	)

	#save("sim_data_1.jld2", sim_data)
end

# ╔═╡ d0875144-8174-4842-ac84-011f6c82f1b1
"""
Simulates the source localization algorithm for multiple starting locations and replicates, returning collected trajectory data for statistical analysis.

For each initial robot position in `robot_starts`, the function runs the localization algorithm `num_replicates` times. Each simulation uses Thompson sampling and MCMC inference to track and approach the radiation source. If a simulation exceeds `max_steps` without locating the source, an error is raised. Results are optionally saved to a JLD2 file.

# arguments
* `rad_sim::RadSim` – The radiation simulation object used to generate counts from a known source field.
* `robot_starts::Vector{Vector{Int64}}` – List of grid index locations where each batch of simulations will begin.

# keyword arguments
* `num_mcmc_samples::Int64=150` – Number of MCMC samples per inference step.
* `num_mcmc_chains::Int64=4` – Number of MCMC chains to run in parallel.
* `I::Float64=I` – Source strength used in the forward model.
* `L::Float64=L` – Length and width of the square domain being explored.
* `Δx::Float64=Δx` – Grid spacing and step size of the robot.
* `allow_overlap::Bool=false` – Whether the robot is allowed to revisit previously visited positions.
* `x₀::Vector{Float64}=[250.0, 250.0]` – True source location; simulation ends when the robot gets within √(2Δx²).
* `z_index::Int=1` – z-slice of the γ-matrix to sample from.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` – Optional list of obstructions (e.g., rectangles or circles) to avoid.
* `exploring_start::Bool=true` – Whether the robot begins with exploratory large steps before converging.
* `num_exploring_start_steps::Int=15` – Initial number of exploratory steps taken in the early phase of the simulation.
* `spiral::Bool=true` – If true, the robot initially explores using a spiral pattern.
* `r_check::Float64=70.0` – Radius (in meters) to check how frequently the robot has sampled locally.
* `r_check_count::Int=10` – Number of samples within `r_check` radius required to trigger coarser movement (i.e., larger step sizes).
* `meas_time::Float64=1.0` – Measurement time (in seconds) per observation.
* `num_replicates::Int64=5` – Number of independent replicates to run per starting location.
* `filename::String="none"` – If not `"none"`, saves all replicate data to a `.jld2` file using the provided filename.

# returns
* `Vector{DataFrame}` – A list of simulation trajectories, one per replicate, each containing time, position, and counts.
"""
function run_batch(
	rad_sim::RadSim, 
	robot_starts::Vector{Vector{Int64}}; 
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	I::Float64=I,
	L::Float64=L,
	Δx::Float64=Δx,
	allow_overlap::Bool=false,
	x₀::Vector{Float64}=[250.0, 250.0],
	z_index::Int=1,
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing,
	exploring_start::Bool=true,
	num_exploring_start_steps::Int=15,
	spiral::Bool=true,
	r_check::Float64=70.0,
	r_check_count::Int=10,
	meas_time::Float64=1.0,
	num_replicates::Int64=5,
	filename::String="none"
)
	test_data = Vector{DataFrame}(undef, num_replicates * length(robot_starts))
	max_steps = 2000
	Turing.setprogress!(false)
	for (i, r_start) in enumerate(robot_starts)
		for j=1:num_replicates
			#run simulation
			index = (i - 1) * num_replicates + j
			test_data[index] = simulate(
				rad_sim,
				max_steps,
				robot_start=r_start, 
				num_mcmc_samples=num_mcmc_samples,
				num_mcmc_chains=num_mcmc_chains,
				I=I,
				L=L,
				Δx=Δx,
				allow_overlap=allow_overlap,
				x₀=x₀,
				save_chains=false,
				z_index=z_index,
				obstructions=obstructions,
				exploring_start=exploring_start,
				num_exploring_start_steps=num_exploring_start_steps,
				spiral=spiral,
				r_check=r_check,
				r_check_count=r_check_count,
				meas_time=meas_time
			)

			if length(test_data[index][:, 1]) >= max_steps
				error("ERROR: case robot start: $(r_start), replicate $(j) unable to find the source after $(max_steps) steps.")
			end
			GC.gc()
		end
	end

	if filename != "none"
		batch_data = Dict("batch" => test_data)
		save("$(filename).jld2", batch_data)
	end
	
	return test_data
end

# ╔═╡ 73bdc00d-58d7-4a04-a880-7b6f1bfc78e8
"""
Runs a grid of source localization simulations across combinations of hyperparameters to evaluate their impact on performance.

This function systematically tests combinations of `exploring_start_steps` and `r_check_vals` using the `run_batch` routine for each pair. Each configuration is evaluated over a set of `robot_starts`, with `num_replicates` simulations per starting point. The results are saved using filenames based on the parameter settings, and overall timing metrics are stored separately for analysis.

⚠️ This function can be computationally expensive. For example, 6 `exploring_start_steps` values × 5 `r_check_vals` × 12 start positions × 100 replicates = 36,000 simulations.

# arguments
* `rad_sim::RadSim` – Radiation simulation object providing the γ-matrix and source behavior.
* `robot_starts::Vector{Vector{Int64}}` – A list of starting positions (as grid indices) for the robot to begin each replicate.

# keyword arguments
* `exploring_start_steps::Vector{Int64}=[20, 17, 15, 12, 10, 5]` – Values for the number of initial exploratory steps to test.
* `r_check_vals::Vector{Float64}=[100.0, 75.0, 50.0, 25.0, 0.0]` – Radii (in meters) to check local sampling density for triggering coarser movement.
* `num_mcmc_samples::Int64=150` – Number of MCMC samples per inference step.
* `num_mcmc_chains::Int64=4` – Number of MCMC chains to run in parallel.
* `I::Float64=I` – Source strength for the forward model.
* `L::Float64=L` – Length/width of the square simulation domain.
* `Δx::Float64=Δx` – Spatial step size for robot movement.
* `allow_overlap::Bool=false` – Whether the robot can revisit previously visited locations.
* `x₀::Vector{Float64}=[250.0, 250.0]` – True source location, used to determine stopping condition.
* `z_index::Int=1` – z-slice of the γ-matrix to simulate from.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` – Optional list of obstructions to include in the simulation environment.
* `r_check_count::Int=10` – Number of samples required within the `r_check` radius to trigger coarser movement.
* `meas_time::Float64=1.0` – Measurement duration per observation.
* `num_replicates::Int64=10` – Number of simulation replicates per configuration and start position.
* `filename::String="batch_1"` – Basename used for output `.jld2` files storing the batch data and timing results.

# returns
* `data_storage::Dict{String, Vector{DataFrame}}` – Dictionary mapping parameter combinations (e.g., `"expl_15_r_50.0"`) to their corresponding batch of simulation replicates.
* `times::Dict{String, Vector{Float64}}` – Dictionary mapping the same parameter combinations to the final timestamps of each replicate in the batch.
"""
function test_params(
	rad_sim::RadSim, 
	robot_starts::Vector{Vector{Int64}}; 
	exploring_start_steps::Vector{Int64}=[20, 17, 15, 12, 10, 5],
	r_check_vals::Vector{Float64}=[100.0, 75.0, 50.0, 25.0, 0.0],
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	I::Float64=I,
	L::Float64=L,
	Δx::Float64=Δx,
	allow_overlap::Bool=false,
	x₀::Vector{Float64}=[250.0, 250.0],
	z_index::Int=1,
	obstructions::Union{Nothing, Vector{Obstruction}}=nothing,
	r_check_count::Int=10,
	meas_time::Float64=1.0,
	num_replicates::Int64=10,
	filename::String="batch_1"
)

	data_storage = Dict()
	times = Dict()

	for (i, expl_step_num) in enumerate(exploring_start_steps)
		for (j, r_val) in enumerate(r_check_vals)
			@info "beginning test: $(expl_step_num) exploring steps and radius value $(r_val)"
			index = (i - 1) * length(r_check_vals) + j
			test_batch = run_batch(
				rad_sim,
				robot_starts,
				num_mcmc_samples=num_mcmc_samples,
				num_mcmc_chains=num_mcmc_chains,
				I=I,
				L=L,
				Δx=Δx,
				allow_overlap=allow_overlap,
				x₀=x₀,
				z_index=z_index,
				obstructions=obstructions,
				exploring_start=true,
				num_exploring_start_steps=expl_step_num,
				spiral=false,
				r_check=r_val,
				r_check_count=r_check_count,
				num_replicates=num_replicates,
				meas_time=meas_time,
				filename="expl_$(expl_step_num)_r_$(r_val)"
			)

			data_storage["expl_$(expl_step_num)_r_$(r_val)"] = test_batch
		end
	end
	save("param_$(filename).jld2", batch_data)

	for (name, batch) in data_storage
		times[name] = [batch[i][end, :]["time"] for i in eachindex(batch)]
	end

	save("param_times_$(filename).jld2", times)
	
	return data_storage, times
end

# ╔═╡ 5ba7c685-8cc2-409d-8ed7-1b2b18cecd89
md"## sample location getter"

# ╔═╡ 5b8c19ce-273a-48e3-9677-f26fb1be9c61
"""
Using Thompson sampling and exploration methods, provide the next location where data should be collected.

# arguments
* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `environment::Environment` - environment struct generated by the `generate_robot_grid_matrix()` function, which should contain a field named grid of type `Array{Union{Bool, Int64}, 3}` where the first two entries are the x and y coordinates of each grid location for the robot search space and the third entry is the matrix of boolean values representing accessibility of locations.
# keyword arguments
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `I::Float64=I` - source strength.
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
* `x₀::Vector{Float64}=[250.0, 250.0]` - source location, this tells the simulation to stop if the current location is within environment.Δx of x₀.
* `save_chains::Bool=false` - set to true to save the MCMC simulation chain data for every step.
* `exploring_start::Bool=true` - if set to false, will begin by taking single steps only towards the greedy choice from Thompson sampling, otherwise will take decreasingly large steps.
* `num_exploring_start_steps::Int=10` - controls the size and number of exploring start steps. For example, a value of 10 will make the robot take 10 steps initially, followed by 9, then 8 etc...
* `r_check::Float64=70.0` - the radius to check for `r_check_count` number of samples to determine if a large next step should be taken.
* `r_check_count::Int=10` - the number of samples within radius `r_check` to determine if a large next step should be taken.
* `disable_log::Bool=true` - set to false to allow logging by Turing.jl.
# returns
* `Vector{Float64}` – A 2D coordinate `[x, y]` representing the next location where the robot should collect data next. The returned position is chosen based on Thompson sampling from the posterior distribution and adjusted for exploration, while ensuring that the location lies within the accessible search grid and avoids obstructions.

"""
function get_next_sample(
	data::DataFrame, 
	environment::Environment;
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	allow_overlap::Bool=false,
	save_chains::Bool=false,
	exploring_start::Bool=true,
	num_exploring_start_steps::Int=10,
	spiral::Bool=false,
	r_check::Float64=70.0,
	r_check_count::Int=10,
	disable_log::Bool=true,
	turn_off_explore_threshold::Real=5
)
	#this is needed for src file where type assertions for structs get weird
	@assert hasfield(typeof(environment), :grid) "environment struct, $(environment), must contain the :grid field."
	
	#pull robot_path from data
	robot_path = [row["x [m]"] for row in eachrow(data)]

	#get min maxes for search space
	L_min = 1.0 * minimum(environment.grid[:, :, 1:2])
	L_max = 1.0 * maximum(environment.grid[:, :, 1:2])

	#establish Poisson model and calc chains
	model = rad_model(data, L_min=L_min, L_max=L_max, environment=environment)
	if disable_log
		Logging.with_logger(NullLogger()) do
			Turing.setprogress!(false)
		model_chain = DataFrame(
			sample(model, NUTS(), MCMCThreads(), num_mcmc_samples, num_mcmc_chains, progress=false)
		)
		end
	else
		model_chain = DataFrame(
			sample(model, NUTS(), MCMCThreads(), num_mcmc_samples, num_mcmc_chains, progress=false)
		)
	end

	#sample posterior and find the next best dir
	next_dir = thompson_sampling(
		robot_path, 
		environment,
		model_chain,
		allow_overlap=allow_overlap
	)

	#establish a single delta, current_pos and
	Δ 			= get_Δ(next_dir, Δx=environment.Δ)
	current_pos = robot_path[end]

	if exploring_start && num_exploring_start_steps - length(robot_path) + 1 > 1
		steps = num_exploring_start_steps - length(robot_path) + 1
	else
		# check how many samples takin within r_check radius
		num_within_r_check = sum(
			norm(pos .- robot_path[end]) ≤ r_check for pos in robot_path
		)
		num_within_r_check -= 1
		# if criteria met, move a lot at once
		steps = num_within_r_check >= r_check_count ? 2 : 1
	end

	#override steps if recent data collection shows high samples
	if data[end, "counts"] > turn_off_explore_threshold
		steps = 1
	end

	#TODO - check from 1 to # steps in next_dir to see if environment.grid[x_coord, y_coord, 3] == true
	for step in 1:steps
        #calc next pos
        next_pos = current_pos .+ step .* Δ
        
        #get grid indicies
        xs = environment.grid[1, :, 1]
        ys = environment.grid[:, 1, 2]
        x_index = argmin(abs.(xs .- next_pos[1]))
        y_index = argmin(abs.(ys .- next_pos[2]))

		#bound check
		if !(0 < x_index ≤ size(environment.grid, 1) && 
			0 < y_index ≤ size(environment.grid, 2))
            @warn "Next position out of bounds. Stopping traversal."
            return save_chains ? (current_pos, model_chain) : current_pos
        end

		#make sure not obstructed
        if !environment.grid[x_index, y_index, 3]
            @warn "Encountered an obstruction at step $(step). Stopping traversal."
            return save_chains ? (current_pos, model_chain) : current_pos
        end

        current_pos = next_pos
    end

	#return final valid pos
	return save_chains ? (current_pos, model_chain) : current_pos
end

# ╔═╡ 28f3b421-6f09-4b52-a032-f67542b1efab
md"## experiment sim"

# ╔═╡ 195b34df-026e-4c02-86c5-7a21c689869f
"""
Runs a source localization simulation in a structured experimental environment defined by a grid.

This function begins at a user-specified grid index within the environment and performs `num_steps` of movement using Thompson sampling-based decision making. It evaluates a probabilistic model at each step, uses MCMC to sample from the posterior, and selects the next location based on sampled target proximity and exploration heuristics. Movement is constrained to accessible grid locations and can optionally avoid previously visited points and obstructions.

# arguments
* `num_steps::Int64` – The number of movement steps the robot should take in the simulation.
* `environment::Environment` – The grid-based environment object. This should contain a field `grid` of type `Array{Union{Bool, Int64}, 3}` with x-coordinates, y-coordinates, and accessibility information.

# keyword arguments
* `robot_start::Vector{Int64}=[41, 43]` – The grid indices (i, j) for the robot's starting location in the environment.
* `num_mcmc_samples::Int64=150` – Number of MCMC samples per inference step.
* `num_mcmc_chains::Int64=4` – Number of MCMC chains to run in parallel.
* `I::Float64=I` – Source strength used in the forward model.
* `allow_overlap::Bool=false` – If `false`, the robot avoids revisiting previously visited locations unless no alternatives exist.
* `x₀::Vector{Float64}=[250.0, 250.0]` – True source location; simulation halts early if the robot comes within `Δx` of this location.
* `save_chains::Bool=false` – If `true`, stores the MCMC chain output at each step for later analysis.
* `exploring_start::Bool=true` – If `true`, the robot starts with large exploratory steps before switching to greedy movement.
* `num_exploring_start_steps::Int=1` – Number of initial large steps before gradually shrinking exploration.
* `spiral::Bool=false` – If `true`, the robot follows a predefined outward spiral pattern at the beginning instead of Thompson sampling.
* `meas_time::Float64=1.0` – Measurement duration per observation.
* `r_check::Float64=70.0` – Radius used to determine local sampling density for adaptive movement scaling.
* `r_check_count::Int=10` – Number of nearby samples within `r_check` required to trigger coarse movement (larger step sizes).
* `disable_log::Bool=true` – If `true`, disables logging output from Turing.jl and MCMC sampling.

# returns
* `sim_data::DataFrame` – A DataFrame containing the simulation results. Each row corresponds to a step in the robot's path and includes the following columns:
  - `"time"` – Cumulative time at each step (including travel and measurement).
  - `"x [m]"` – 2D position of the robot in meters.
  - `"counts"` – Measured radiation counts at each location.

* `sim_chains::Dict{Int, DataFrame}` *(only if `save_chains=true`)* – A dictionary mapping each simulation step index to the corresponding MCMC chain output as a DataFrame. Each chain represents the posterior samples for source location and intensity at that step.
"""
function sim_exp(
    num_steps::Int64,
    environment::Environment;
    robot_start::Vector{Int64}=[41, 43],
    num_mcmc_samples::Int64=150,
    num_mcmc_chains::Int64=4,
    I::Float64=I,
    allow_overlap::Bool=false,
    x₀::Vector{Float64}=[250.0, 250.0],
    save_chains::Bool=false,
    exploring_start::Bool=true,
    num_exploring_start_steps::Int=1,
    spiral::Bool=false,
    meas_time::Float64=1.0,
    r_check::Float64=70.0,
    r_check_count::Int=10,
    disable_log::Bool=true
)
    @assert hasfield(typeof(environment), :grid) "environment struct must contain the :grid field."

    # Initialize robot path and starting sample
    x_start = 1.0 * environment.grid[robot_start[2], robot_start[1], 1]
    y_start = 1.0 * environment.grid[robot_start[2], robot_start[1], 2]
    robot_path = [[x_start, y_start]]

    sample_mean = mean(count_Poisson(robot_path[end], x₀, I))
    noise = rand(Poisson(0.5)) * rand([-1, 1])
    start_sample = round(Int, max(sample_mean + noise, 0))

    sim_data = DataFrame(
        "time" => [meas_time],
        "x [m]" => [robot_path[end]],
        "counts" => [start_sample]
    )
    sim_chains = Dict{Int, DataFrame}()

    for step = 1:num_steps

        result = get_next_sample(
            sim_data,
            environment;
            num_mcmc_samples=num_mcmc_samples,
            num_mcmc_chains=num_mcmc_chains,
            allow_overlap=allow_overlap,
            save_chains=save_chains,
            exploring_start=exploring_start,
            num_exploring_start_steps=num_exploring_start_steps,
            spiral=spiral,
            r_check=r_check,
            r_check_count=r_check_count,
            disable_log=disable_log
        )

		if norm(robot_path[end] .- x₀) <= (2 * environment.Δ^2)^(0.5)
            @info "Source found at step $(step), robot at location $(robot_path[end])"
			sim_chains[step] = result[2]
            break
        end

	if save_chains
		if result isa Tuple{Vector{Float64}, DataFrame}
			next_pos, model_chain = result
			sim_chains[step] = model_chain
		else
			@warn "Expected tuple return but got $(typeof(result)). Using result as next_pos."
			next_pos = result
		end
	else
		next_pos = result
	end

        Δt_travel = norm(next_pos .- robot_path[end]) / r_velocity
        robot_path = push!(robot_path, next_pos)

        sample_mean = mean(count_Poisson(next_pos, x₀, I))
        noise = rand(Poisson(λ_background)) * rand([-1, 1])
        counts = round(Int, max(sample_mean + noise, 0))

        push!(sim_data, Dict(
            "time" => sim_data[end, "time"] + Δt_travel + meas_time,
            "x [m]" => next_pos,
            "counts" => counts
        ))
    end

    return save_chains ? (sim_data, sim_chains) : sim_data
end


# ╔═╡ 7647103a-27fe-436a-87cf-301b52195174
exp_test, exp_chains = sim_exp(500, grid_env, save_chains=true)

# ╔═╡ 4b0e6a4c-c6b8-4347-b960-8f9b5ff318d4
exp_test

# ╔═╡ 3277b87e-f8b4-49cb-bcb6-50ffb91f2145
get_next_sample(exp_test, grid_env)

# ╔═╡ 4e9e816b-dae3-4a6e-9453-f925ac70f140
#= #UNCOMMENT TO SAVE PICTURES FOR EVERY STEP OF THE EXPERIMENTAL SIMULATION
begin
	for i=1:length(exp_chains)
		fig = viz_robot_grid(grid_env, data_collection=exp_test[1:i, :], chain_data=exp_chains[i], show_grid=false, x₀=[250.0, 250.0])
		save("exp_j$(i).png", fig)
	end
end
=#

# ╔═╡ 11df326c-8cda-4075-be87-c96d94baaec2
viz_robot_grid(grid_env, data_collection=exp_test[1:end, :], chain_data=exp_chains[length(exp_chains)-1], show_grid=false, x₀=[250.0, 250.0], view_chain_as_hm=false)

# ╔═╡ e5ba01be-75c3-4f64-959d-bcdf3f49a8cb
viz_num=30

# ╔═╡ 7ec5c32b-a459-4af1-b056-5ce81acab80b
viz_robot_grid(grid_env, data_collection=exp_test[1:end, :], chain_data=exp_chains[viz_num], show_grid=false, x₀=[250.0, 250.0], view_chain_as_hm=true)

# ╔═╡ 1699ebaf-8781-4c0b-a472-66ea0710770e
data

# ╔═╡ 23bf5985-ad58-447a-94e1-5dcde90e358e
exp_test[1:end, :]

# ╔═╡ a41efa84-556b-47ac-a9b1-777d5d453686
md"""
# TODO
"""

# ╔═╡ f77e064c-c86c-4a75-9d93-7618d309e308
"""
Loads a CSV file containing raw sensor data from the vacuum robot, removes the header, and standardizes the format to match internal data structures.

# arguments
* `data_path::String`: Path to the CSV file. The file is expected to have a header row followed by columns representing time, x-position, y-position, and count measurements.

# returns
* `DataFrame`: A standardized DataFrame with the following columns:
  * `"time"`: Time of measurement (from column 1 of the CSV).
  * `"x [m]"`: A 2-element vector `[x, y]` of spatial coordinates (from columns 2 and 3).
  * `"counts"`: Measurement count rounded to the nearest integer (from column 4).
"""
function load_and_standardize_data_from_csv(data_path::String)
    #load CSV, skipping the first header row
    raw_df = CSV.read(data_path, DataFrame; header=false, skipto=2)

    #standardized DataFrame
    df = DataFrame(
        "time" => raw_df[:, 1],
        "x [m]" => [ [x, y] for (x, y) in zip(raw_df[:, 2], raw_df[:, 3]) ],
        "counts" => round.(Int, raw_df[:, 4])
    )

    return df
end

# ╔═╡ 15c3757a-8f94-4070-989e-3725d73d4495
load_and_standardize_data_from_csv(joinpath(pwd(), joinpath("csv", "Collected_data.csv")))

# ╔═╡ 728b148f-1783-406a-9e76-ce63f7fa2707
readdir("csv")

# ╔═╡ b5e8d79d-e8ad-4778-90e3-50c838053c1f
md"""
write a function that reads CSV file to extract data being populated by python script.

* I need csv file with data.

That function will in turn create/overwrite a csv with the next position to collect data.

this function needs to internally find walls.csv in the folder and create the environment struct.
"""

# ╔═╡ 27f0209f-6b74-4e48-b6b0-e9f729d44308
"""
This function performs 3 major tasks, it uploads the data file from the collected_data.csv path to inform the inference engine, it uploads the environment from the walls.csv path and generates the environment struct, it then uses Thompson sampling and exploration methods to provide the next location where data should be collected within the masked environment. The next sample location is added to a csv file.

# arguments
* `data_path::String` - the path to the collected_data.csv file containing data showing where the robot has collected data thus far with the last entry being its current location.
* `walls_path::String` - the path to the walls.csv file containing data generated by the vaacuum robot containing only 0's and 1's where 0's represent empty space and 1's represent obstructions.
# keyword arguments
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `I::Float64=I` - source strength.
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
* `x₀::Vector{Float64}=[250.0, 250.0]` - source location, this tells the simulation to stop if the current location is within environment.Δx of x₀.
* `save_chains::Bool=false` - set to true to save the MCMC simulation chain data for every step.
* `exploring_start::Bool=true` - if set to false, will begin by taking single steps only towards the greedy choice from Thompson sampling, otherwise will take decreasingly large steps.
* `num_exploring_start_steps::Int=10` - controls the size and number of exploring start steps. For example, a value of 10 will make the robot take 10 steps initially, followed by 9, then 8 etc...
* `r_check::Float64=70.0` - the radius to check for `r_check_count` number of samples to determine if a large next step should be taken.
* `r_check_count::Int=10` - the number of samples within radius `r_check` to determine if a large next step should be taken.
* `disable_log::Bool=true` - set to false to allow logging by Turing.jl.
# returns
* `Vector{Float64}` – A 2D coordinate `[x, y]` representing the next location where the robot should collect data next. The returned position is chosen based on Thompson sampling from the posterior distribution and adjusted for exploration, while ensuring that the location lies within the accessible search grid and avoids obstructions.
* `mask_clearance::Int=5` - The thickness of walls, larger values will ensure small openings are blocked, set to 0 to not use a mask at all.
* `Δ::Int` - The size of each step in the robot exploration matrix. For example, a step of 10 means each index in the robot matrix is 10 times further apart then the original obstruction map.

"""
function get_update_next_pos(
	data_path::String, 
	walls_path::String;
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	allow_overlap::Bool=false,
	save_chains::Bool=false,
	exploring_start::Bool=true,
	num_exploring_start_steps::Int=10,
	r_check::Float64=70.0,
	r_check_count::Int=10,
	disable_log::Bool=true,
	turn_off_explore_threshold::Real=7,
	seed_loc::Tuple{Int, Int}=(250, 250),
	mask_clearance::Int=5,
	Δ::Int=10
)

	data = LoadData.load_and_standardize_data_from_csv(data_path)

	#load vacuum env
	vac_environment = LoadData.parse_numpy_csv_file(walls_path)
	#mask environment by thickening walls and removing empty wall space
	environment_masked = LoadData.flood_fill(
		vac_environment, 
		seed_loc, 
		clearance_radius=mask_clearance
	)
	#load into environment struct and build robot test matrix
	grid_env =  generate_robot_grid_matrix(vac_environment, Δ)


	next_pos = get_next_sample(
		data, 
		grid_env,
		num_mcmc_samples=num_mcmc_samples,
		num_mcmc_chains=num_mcmc_chains,
		allow_overlap=allow_overlap,
		save_chains=save_chains,
		exploring_start=exploring_start,
		num_exploring_start_steps=num_exploring_start_steps,
		r_check=r_check,
		r_check_count=r_check_count,
		disable_log=disable_log,
		turn_off_explore_threshold=turn_off_explore_threshold
	)

	#save next pos to csv
    open(joinpath("csv","next_pos.csv"), "w") do io
        writedlm(io, [next_pos'], ',')
    end

end

# ╔═╡ 107eff0d-daee-4b09-9eb6-e56ec6c4a5b2
get_update_next_pos(joinpath("csv", "Collected_data.csv"), joinpath("csv", "walls.csv"))

# ╔═╡ d5364993-9004-41bc-886e-49a7c1830461
typeof((250, 250))

# ╔═╡ Cell order:
# ╠═285d575a-ad5d-401b-a8b1-c5325e1d27e9
# ╠═4358f533-5944-406f-80a1-c08f99610d5b
# ╠═54b50777-cfd7-43a3-bcc2-be47f117e635
# ╠═a03021d8-8de2-4c38-824d-8e0cb571b9f1
# ╟─2d9fc926-9ed0-4a29-8f5c-d57a7b2300fe
# ╠═64830991-da9b-4d5d-a0db-3030d9461a8d
# ╟─52d76437-0d60-4589-996f-461eecf0d45d
# ╟─e5b1369e-0b4b-4da3-9c95-07ceda11b31d
# ╠═f639c03d-bdc3-43e5-b864-3277bbf02273
# ╠═064eb92e-5ff0-436a-8a2b-4a233ca4fa42
# ╠═dd15ee55-76cd-4d56-b4a6-aa46212c176b
# ╠═b8d6c195-d639-4438-8cab-4dcd99ea2547
# ╟─82577bab-5ce9-4164-84db-9cfa28b501b0
# ╟─a9c65b66-eb92-4943-91a9-9d0ea6cfa3d3
# ╠═57478c44-578e-4b53-b656-c9b3c208dec4
# ╟─03910612-d3fe-481c-bb70-dd5578bd8258
# ╠═dd357479-64ef-4823-8aba-931323e89aed
# ╟─aca647fb-d8c4-4966-a641-3b361295f1e2
# ╠═2eba07e7-7379-4b2d-bad3-a6f1d52b676e
# ╟─7fcecc0e-f97c-47f7-98db-0da6d6c1811e
# ╟─9ac1e08a-cd89-4afb-92d9-2bd973f06aaa
# ╠═e80af66e-54ab-4661-bd49-89328c17e3d4
# ╠═2c959785-6d71-49c6-921a-16e74cb3b43e
# ╠═0fc694a6-f4cf-478d-bd68-9af5f7f4f5b8
# ╟─181c27f4-6830-4c4d-9392-3237564e6cb1
# ╟─7c44611e-2442-4dca-9624-e18676e0f67c
# ╠═52814746-ae35-4ffa-9be0-66854a4d96bf
# ╠═a092d5de-8828-4fa6-8ef5-fb0838cc0887
# ╠═e62ba8da-663a-4b58-afe6-910710d7518e
# ╠═19c95a83-670c-4ad6-82a1-5a4b6809f1d4
# ╟─8ffaf344-1c74-48c8-a116-8c937322cd6e
# ╠═1197e64f-34c2-4892-8da5-3b26ee6e7c2f
# ╟─7278adb5-2da1-4ea1-aa38-d82c23510242
# ╠═63c8b6dd-d12a-42ec-ab98-1a7c6a991dbd
# ╠═173feaf5-cbfa-4e94-8de5-1a5311cdf14e
# ╠═7211ea6e-6535-4e22-a2ef-a1994e81d22a
# ╠═1f12fcd1-f962-45e9-8c07-4f42f869d6a0
# ╠═bbeed200-3315-4546-9540-16864843f178
# ╟─3ee48a88-3b6a-4995-814b-507679065ff4
# ╠═6fdb31a6-f287-40a4-8c21-2ef92ff90a99
# ╠═ca763f28-d2b2-4eac-ae6c-90f74e3c42e7
# ╠═00f909f7-86fd-4d8e-8db2-6a587ba5f12d
# ╠═e2387282-0c5e-4115-9868-738c864ce41b
# ╠═5e7dc22c-dc2d-41b0-bb3e-5583a5b79bdd
# ╠═4bda387f-4130-419c-b9a5-73ffdcc184f9
# ╟─560de084-b20b-45d7-816a-c0815f398e6d
# ╠═6d4e86ae-701c-443e-9d05-0cc123adbc29
# ╠═45014b50-c04b-4f42-83c3-775ec6cd6e3f
# ╠═66d5216d-66a8-4e33-9f36-54a0d4ec4459
# ╠═b48a8a94-7791-4e10-9a6b-1c6f2eca7968
# ╠═b63d1384-02ef-45c6-80c8-fdfd54ce6804
# ╠═d47b2021-7129-4a31-8585-2c7257489b1a
# ╟─849ef8ce-4562-4353-8ee5-75d28b1ac929
# ╠═e622cacd-c63f-416a-a4ab-71ba9d593cc8
# ╟─8ed5d321-3992-40db-8a2e-85abc3aaeea0
# ╠═6fa37ac5-fbc2-43c0-9d03-2d194e136951
# ╠═0175ede7-b2ab-4ffd-8da1-278120591027
# ╠═5b9aaaeb-dbb3-4392-a3a1-ccee94d75fed
# ╟─31864185-6eeb-4260-aa77-c3e94e467558
# ╟─015b9f4d-09b8-49f3-bc03-2fd3b972e933
# ╠═126df6ec-9074-4712-b038-9371ebdbc51d
# ╟─bfe17543-7b54-4f52-9679-f723adafdbdd
# ╠═83052e75-db08-4e0a-8c77-35487c612dae
# ╠═c6c39d74-4620-43df-8eb1-83c436924530
# ╠═3c0f8b63-6276-488c-a376-d5c554a5555d
# ╠═ddc23919-17a7-4c78-86f0-226e4d447dbe
# ╠═50e623c0-49f6-4bb5-9b15-c0632c3a88fd
# ╟─de738002-3e80-4aab-bedb-f08533231ed7
# ╠═dac856e6-f651-49d0-994e-46c8296a3d30
# ╠═deae0547-2d42-4fbc-b3a9-2757fcfecbaa
# ╠═82425768-02ba-4fe3-ab89-9ac95a45e55e
# ╠═9fbe820c-7066-40b5-9617-44ae0913928e
# ╟─50832f87-c7eb-4418-9864-0f807a16e7a7
# ╠═7d0e24e2-de5b-448c-8884-4d407ead1319
# ╠═22652624-e2b7-48e9-bfa4-8a9473568f9d
# ╠═ec9e4693-771c-467d-86cc-ab2ba90019fe
# ╟─3ae4c315-a9fa-48bf-9459-4b7131f5e2eb
# ╟─c6783f2e-d826-490f-93f5-3da7e2717a02
# ╠═1e7e4bad-16a0-40ee-b751-b2f3664f6620
# ╠═13ff8f6a-7bb2-41a0-83ac-7c9fca962605
# ╟─21486862-b3c2-4fcc-98b2-737dcc5211fb
# ╠═2fe974fb-9e0b-4c5c-9a5a-a5c0ce0af065
# ╠═2b656540-9228-4d8a-9db1-1c0bec3d33f3
# ╠═0a39daaa-2c20-471d-bee3-dcc06554cf78
# ╠═a8281a0d-2b9c-40d7-802a-ca434ba602f9
# ╠═9dd28728-f8de-43bb-8e0c-e7384f924adc
# ╠═ea2dc60f-0ec1-4371-97f5-bf1e90888bcb
# ╟─65d603f4-4ef6-4dff-92c1-d6eef535e67e
# ╠═10c62123-4ae4-4688-87e1-9b0397c75e88
# ╠═8853858d-1d47-40c9-94a4-c912ad00af5d
# ╟─aa72cf61-839d-4707-95c8-0a9230e77d56
# ╠═f4d234f9-70af-4a89-9a57-cbc524ec52b4
# ╠═4bb02313-f48b-463e-a5b6-5b40fba57e81
# ╟─95837cad-192d-46b4-aaa4-c86e9b1d1c09
# ╟─8c18d4e8-fd5c-4fd4-8f1e-516615a9e5f0
# ╠═ed8381e3-7a83-4afc-a95d-5cbd03b7e852
# ╟─eafb66bc-6da3-4570-b62a-922627e6ccde
# ╠═9d0795fa-703e-47a4-8f1e-fe38b9d604b4
# ╠═600a5f36-cfa2-4848-8984-44f0ae54ed67
# ╠═b7673342-70f2-4cbb-869e-1b67f9ee7235
# ╠═c6b9ca97-7e83-4703-abb9-3fd43daeb9a7
# ╠═28f9219a-6758-4aa3-8f96-f5b2e8a8e5d7
# ╠═97de1cb8-9c72-440b-896a-a1f1d24e46f5
# ╠═f063123b-bab8-435c-b128-0dc72d31b5fb
# ╟─bb94ce77-d48c-4f6d-b282-96197d6e7b6b
# ╟─f55544f3-413d-44c5-8e81-37a5f017b460
# ╠═a2154322-23de-49a6-9ee7-2e8e33f8d10c
# ╠═8b98d613-bf62-4b2e-9bda-14bbf0de6e99
# ╠═161833bd-aa39-4bab-98b6-be10b6d3653f
# ╟─76bc6fcb-0018-40dd-9709-65bf9d223615
# ╠═12729385-04ea-4c8b-ab08-e114b4f4172d
# ╠═32228db5-bf76-4633-ab5e-224f95459cc9
# ╟─ff90c961-70df-478a-9537-5b48a3ccbd5a
# ╠═52296a3f-9fad-46a8-9894-c84eb5cc86d7
# ╟─44d81172-2aef-4ef1-90e9-6a169e92f9ff
# ╠═ae92f6ae-298d-446d-b379-ee2190ef1915
# ╟─ad54d1fa-b3e7-4aeb-96f4-b5d15dee38d5
# ╠═5c03dc8e-6484-4a73-8cb7-eb43aa382a9d
# ╠═f847ac3c-6b3a-44d3-a774-4f4f2c9a195d
# ╠═e7023831-5c03-4f53-95f4-ab837bced1b2
# ╠═22a012c1-4169-4959-af47-9d4b01691ae9
# ╠═9a1fa610-054b-4b05-a32b-610f72329166
# ╠═f5ea3486-4930-42c2-af1b-d4a17053976a
# ╠═4a0c8aab-2424-441d-a8c7-9f8076ecbae7
# ╟─2f7a3d49-1864-4113-b173-ee7e8c9e62a4
# ╠═ef7ff4ec-74ac-40b9-b68b-dbc508e50bef
# ╠═ac2dd9e7-0547-4cda-acf5-845d12d87626
# ╠═3ff17eaf-974d-4bf0-b75f-d3ef473730bf
# ╠═ea505dc1-a18f-408f-bff8-3b488c49fdb0
# ╠═d14fc2b4-ad11-4506-a580-06bfefede40b
# ╟─a53b3039-eb9e-45aa-914f-034d2a5b6e01
# ╠═34527801-4098-4ffe-99c0-5abbdd99ee55
# ╟─40cfe92d-b707-4b22-b3f9-228e5a0df7b2
# ╟─0c2d090c-82c8-466d-aea7-140b4422d254
# ╠═5e5c4e18-63b9-4b2b-bf75-52c77ec3d0fe
# ╠═fd3393e0-9e08-41e6-a6d2-c28743cb1a68
# ╟─e75b8aae-8da8-45e8-8405-103f77a3cca6
# ╠═d0875144-8174-4842-ac84-011f6c82f1b1
# ╠═73bdc00d-58d7-4a04-a880-7b6f1bfc78e8
# ╠═c84fc2c3-5d44-49dd-a176-cf7277b4ef30
# ╠═e5ead52b-c407-400d-9a26-fca9b61556f3
# ╠═5e5bf646-0a05-4405-8563-86abe65d6fca
# ╟─75cda12e-3a12-44b4-aa51-ef60588fee49
# ╠═de432aa4-b320-4598-aedd-32ca2b74be52
# ╟─1c0fee71-29a2-4fa1-93db-e1213ed88bb0
# ╠═54e52416-6c81-4dae-be10-2ddd1449dbfa
# ╟─5ba7c685-8cc2-409d-8ed7-1b2b18cecd89
# ╠═5b8c19ce-273a-48e3-9677-f26fb1be9c61
# ╟─28f3b421-6f09-4b52-a032-f67542b1efab
# ╠═195b34df-026e-4c02-86c5-7a21c689869f
# ╠═7647103a-27fe-436a-87cf-301b52195174
# ╠═4b0e6a4c-c6b8-4347-b960-8f9b5ff318d4
# ╠═3277b87e-f8b4-49cb-bcb6-50ffb91f2145
# ╠═4e9e816b-dae3-4a6e-9453-f925ac70f140
# ╠═11df326c-8cda-4075-be87-c96d94baaec2
# ╠═e5ba01be-75c3-4f64-959d-bcdf3f49a8cb
# ╠═7ec5c32b-a459-4af1-b056-5ce81acab80b
# ╠═1699ebaf-8781-4c0b-a472-66ea0710770e
# ╠═23bf5985-ad58-447a-94e1-5dcde90e358e
# ╟─a41efa84-556b-47ac-a9b1-777d5d453686
# ╠═f77e064c-c86c-4a75-9d93-7618d309e308
# ╠═15c3757a-8f94-4070-989e-3725d73d4495
# ╠═728b148f-1783-406a-9e76-ce63f7fa2707
# ╟─b5e8d79d-e8ad-4778-90e3-50c838053c1f
# ╠═27f0209f-6b74-4e48-b6b0-e9f729d44308
# ╠═107eff0d-daee-4b09-9eb6-e56ec6c4a5b2
# ╠═d5364993-9004-41bc-886e-49a7c1830461

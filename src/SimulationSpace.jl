

module SimulationSpace
include("Constants.jl")
include("RadModelStructs.jl")
include("InferenceEngine.jl")

using .InferenceEngine, .Constants, .RadModelStructs, LinearAlgebra, Turing, SpecialFunctions, DataFrames, StatsBase, Distributions, JLD2, Logging, LatinHypercubeSampling

#############################################################################
##  EXAMPLES
#############################################################################

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

#############################################################################
##  MOVE & SAMPLE MODEL
#############################################################################
"""
Moves the robot one step in the specified direction according to the grid spacing `Δx`.

This function appends a new position to the robot's path based on the direction and step size. It does not check for obstructions or boundary limits.

# arguments
* `robot_path::Vector{Vector{Float64}}` - The robot's current path, where the last element is the current position.
* `direction::Symbol` - Direction to move. Must be one of `:up`, `:down`, `:left`, or `:right`.

# keyword arguments
* `Δx::Float64=Δx` - The grid spacing value (movement step size).

# modifies
* `robot_path` - Updated in-place with one additional position in the given direction.
"""

function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol; Δx::Float64=Δx)
	Δ = get_Δ(direction, Δx=Δx)
	push!(robot_path, robot_path[end] + Δ)
end

"""
Moves the robot `n` times in a specified direction by modifying the robot path in-place.

This function either performs `n` individual steps or one aggregated movement of `n × Δx`, depending on the `one_step` flag. Movement halts early if it would result in collision with an obstruction or leave the defined environment bounds.

# arguments
* `robot_path::Vector{Vector{Float64}}` - The robot's current path, with the last element representing its current position.
* `direction::Symbol` - The direction to move. Must be one of `:up`, `:down`, `:left`, or `:right`.
* `n::Int` - The number of steps to move.

# keyword arguments
* `Δx::Float64=Δx` - The grid spacing value.
* `one_step::Bool=false` - If `true`, moves the robot `n` steps in one large movement. If `false`, moves one step at a time, allowing for intermediate checks or measurements.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - Optional list of obstructions; movement will halt if any are encountered.
* `L::Float64=1000.0` - The length of the square environment; movement outside this boundary is not allowed.

# modifies
* `robot_path` - Updated in-place with the new robot position(s), depending on movement conditions and the `one_step` setting.
"""

function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol, n::Int; Δx::Float64=Δx, one_step::Bool=false, obstructions=nothing, L::Float64=1000.0)
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

"""
Given the grid spacing, provides an index given the provided position vector.

# arguments
* `pos::Vector{Float64}` - current position for which you want the corresponding index.
# keyword arguments
* `Δx::Float64=10.0` - grid spacing.
# returns
* `Tuple{Int, Int}` – A tuple `(i, j)` representing the discrete grid indices corresponding to the input position `pos`. The indices are 1-based and computed by flooring the position divided by the grid spacing `Δx`. This maps continuous coordinates to matrix-style indexing.
"""
function pos_to_index(pos::Vector{Float64}; Δx::Float64=Δx)
    x₁ = Int(floor((pos[1]) / Δx)) + 1
    x₂ = Int(floor((pos[2]) / Δx)) + 1
    return (x₁, x₂)
end

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

"""
Given the current position and radiation simulation model, samples the model by pulling the value from the radiation field and adding noise.

This function retrieves the expected count rate at position `x` from the radiation simulation and adds Poisson-distributed background noise. It is used to simulate what a sensor would detect at a given location.

# arguments
* `x::Vector{Float64}` - The current position `[x, y]` at which the model is sampled.
* `rad_sim::RadSim` - The radiation simulation object containing gamma flux data and metadata.

# keyword arguments
* `I::Float64=I` - Source strength (emissions per second).
* `Δx::Float64=Δx` - Spatial discretization (grid resolution in meters).
* `z_index::Int=1` - Index of the z-slice in the 3D simulation; `1` corresponds to the ground plane.

# returns
* `Int` - Simulated measured count value (non-negative), computed from the expected gamma flux at the given position and perturbed with background noise.
"""
function sample_model(x::Vector{Float64}, rad_sim; I::Float64=Constants.I, Δx::Float64=Δx, z_index::Int=1)
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

#############################################################################
##  NEXT STEP CHECKERS
#############################################################################
"""
Checks whether a 2D position lies within a geometric obstruction.

This function determines whether a point `pos` lies inside a given shape. It currently supports:
- `Rectangle` objects, defined by center, width, and height (axis-aligned),
- `Circle` objects, defined by center and radius.

# arguments
* `pos::Vector{Float64}` - A 2D position `[x, y]` to check for overlap.
* `shape` - An obstruction, either a `Rectangle` or `Circle`, with appropriate geometric parameters.

# returns
* `Bool` - `true` if `pos` lies within the specified shape, `false` otherwise.
"""
function overlaps(pos::Vector{Float64}, shape)

    if hasfield(typeof(shape), :radius)
        x, y = pos
        cx, cy = shape.center
        return (x - cx)^2 + (y - cy)^2 ≤ shape.radius^2
    elseif hasfield(typeof(shape), :width) && hasfield(typeof(shape), :height)
        x, y = pos
        cx, cy = shape.center
        hw, hh = shape.width / 2, shape.height / 2
        return (cx - hw ≤ x ≤ cx + hw) && (cy - hh ≤ y ≤ cy + hh)
    else
        return error("unknown shape type")
    end
end


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
	obstructions=nothing
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

#############################################################################
##  THOMPSON SAMPLING
#############################################################################
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
	obstructions=nothing)

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

#############################################################################
##  SIMULATION AND BATCH PARAMETER EVALUATION
#############################################################################
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
	obstructions=nothing
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
	rad_sim,
	num_steps::Int64; 
	robot_start::Vector{Int64}=[0, 0], 
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	I::Float64=Constants.I,
	L::Float64=L,
	Δx::Float64=Δx,
	allow_overlap::Bool=false,
	x₀::Vector{Float64}=x₀,
	save_chains::Bool=false,
	z_index::Int=1,
	obstructions=nothing,
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
		#TODO - ADD SPIRAL FUNCTIONALITY
		error("SPIRAL FUNCTIONALITY NOT YET IMPLEMENTED HERE!")
		#spiral_control = init_spiral(copy(robot_path[end]), step_init=1, step_incr=2)
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
	rad_sim, 
	robot_starts::Vector{Vector{Int64}}; 
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	I::Float64=Constants.I,
	L::Float64=L,
	Δx::Float64=Δx,
	allow_overlap::Bool=false,
	x₀::Vector{Float64}=[250.0, 250.0],
	z_index::Int=1,
	obstructions=nothing,
	exploring_start::Bool=true,
	num_exploring_start_steps::Int=15,
	spiral::Bool=false,
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

    #save data if a filename is selected
	if filename != "none"
		batch_data = Dict("batch" => test_data)
		save("$(filename).jld2", batch_data)
	end
	
	return test_data
end

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
	rad_sim, 
	robot_starts::Vector{Vector{Int64}}; 
	exploring_start_steps::Vector{Int64}=[20, 17, 15, 12, 10, 5],
	r_check_vals::Vector{Float64}=[100.0, 75.0, 50.0, 25.0, 0.0],
	num_mcmc_samples::Int64=150,
	num_mcmc_chains::Int64=4,
	I::Float64=Constants.I,
	L::Float64=L,
	Δx::Float64=Δx,
	allow_overlap::Bool=false,
	x₀::Vector{Float64}=[250.0, 250.0],
	z_index::Int=1,
	obstructions=nothing,
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
end
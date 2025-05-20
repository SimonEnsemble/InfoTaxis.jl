

module SimulationSpace
include("Constants.jl")
include("RadModelStructs.jl")
include("InferenceEngine.jl")

using .InferenceEngine, .Constants, .RadModelStructs, LinearAlgebra, Turing, SpecialFunctions, DataFrames, StatsBase, Distributions, JLD2, Logging, LatinHypercubeSampling

#############################################################################
##  MOVE & SAMPLE MODEL
#############################################################################
"""
Moves the robot once in the direction provided according to the `Δx` spacing value.

* `robot_path::Vector{Vector{Float64}}` - current robot path.
* `direction::Symbol` - direction to move, must be :up, :down, :left, or :right.

* `Δx::Float64=Δx` - grid spacing value.
"""
function move!(robot_path::Vector{Vector{Float64}}, direction::Symbol; Δx::Float64=Δx)
	Δ = get_Δ(direction, Δx=Δx)
	push!(robot_path, robot_path[end] + Δ)
end

"""
Moves the robot `n` times in a single direction by altering the robot path.

* `robot_path::Vector{Vector{Float64}}` - current robot path.
* `direction::Symbol` - direction to move, must be :up, :down, :left, or :right.
* `n::Int` - the number of times to move.

* `Δx::Float64=Δx` - grid spacing value.
* `one_step::Bool=false` - set to true to move n spaces in one big step (instead of stoping at each Δx to collect data)
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - the vector of obstruction objects.
* `L::Float64=1000.0` - size of the grid space.
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

* `pos::Vector{Float64}` - current position for which you want the corresponding index.
* `Δx::Float64=10.0` - grid spacing.
"""
function pos_to_index(pos::Vector{Float64}; Δx::Float64=Δx)
    x₁ = Int(floor((pos[1]) / Δx)) + 1
    x₂ = Int(floor((pos[2]) / Δx)) + 1
    return (x₁, x₂)
end

"""
Converts a direction to a vector represnetation of movement.

* `direction::Symbol` - :up, :down, :left, or :right
* `Δx::Float64=Δx` - grid spacing value.
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
Given the current position and radiation simulation model, samples the model by pulling the value from the radiation simulation and adding some noise.

* `x::Vector{Float64}` - current position for which you are sampling the model.
* `rad_sim` - the radiation simulation RadSim struct containing the simulation data.
* `I::Float64=I` - source strength.
* `Δx::Float64=Δx` - grid spacing.
* `z_index::Int=1` - 1 is the ground floor index of the set of 2-D simulation slices.
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
Checker if current position overlaps with an obstruction object.
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

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=10.0` - step size of the robot.
* `allow_overlap::Bool=false` - if set to true, allows the robot to backtrack over the previously visited position.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - vector of obstruction objects, currently only accepting Rectangle and Circle types.
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

* `robot_path::Vector{Vector{Float64}}` - the path the robot has taken thus far with the last entry being its current location.
* `pr_field::Matrix{Float64}` - current posterior for the source location.
* `chain::DataFrame` - MCMC test data, this will be used feed concentration values from the forward model into a new MCMC test simulations to arrive at a posterior from which we calculate the entropy.
* `num_mcmc_samples::Int64=100` - the number of MCMC samples per simulation.
* `num_mcmc_chains::Int64=1` - the number of chains of MCMC simulations.
* `L::Float64` - the width/length of the space being explored.
* `Δx::Float64=2.0` - step size of the robot.
* `allow_overlap::Bool=false` - allow the algorithm to overlap over previously visited locations, If set to false, it will only visit previously visited locations in the case where it has no other choice.
* `obstructions::Union{Nothing, Vector{Obstruction}}=nothing` - vector of obstruction objects, currently only accepting Rectangle and Circle types.
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

* `num_samples::Int=15` - number of sample start locations.
* `L::Float64=L` - space size.
* `Δx::Float64=Δx` - discretization.
* `obstructions=nothing` - vector of obstruction objects, if a starting location ends up inside an obstruction, will resample with the latin hypercube options selected.
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

* `rad_sim::RadSim` - the radiation simulation RadSim to sample from.
* `num_steps::Int64` - set the max number of steps to simulate movement.
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


	if save_chains
		return sim_data, sim_chains
	else
		return sim_data
	end
end

"""
Simulates the source localization algorithm several times and collects statistical data.
"""
function run_batch(
	rad_sim::RadSim, 
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

    #save data if a filename is selected
	if filename != "none"
		batch_data = Dict("batch" => test_data)
		save("$(filename).jld2", batch_data)
	end
	
	return test_data
end

"""
Test a series of hyperparameters using batch mode over latin hypercube sampled start locations replicated `num_replicates` times.

This will take a long time as there will be a lot of simulations. For example, with 100 replicates, 6 different `exploring_start_steps` options, 5 `r_check` value options, and 12 latin hypercube sample start locations... will end up running simulate 100 * 6 * 5 * 12 = 36000 times.

each batch will be saved and named by the parameter values.
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
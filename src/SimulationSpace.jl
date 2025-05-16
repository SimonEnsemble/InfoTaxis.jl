

module SimulationSpace
include("Constants.jl")
include("RadModelStructs.jl")
include("InferenceEngine.jl")

using .InferenceEngine, .Constants, .RadModelStructs, LinearAlgebra, Turing, SpecialFunctions, DataFrames, StatsBase, Distributions, JLD2, Logging, LatinHypercubeSampling

#############################################################################
##  MOVE & SAMPLE MODEL
#############################################################################

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
Given the current position and radiation simulation model, samples the model by pulling the value from the radiation simulation and adding some noise.

* `x::Vector{Float64}` - current position for which you are sampling the model.
* `rad_sim` - the radiation simulation RadSim struct containing the simulation data.
* `I::Float64=I` - source strength.
* `Δx::Float64=Δx` - grid spacing.
* `z_index::Int=1` - 1 is the ground floor index of the set of 2-D simulation slices.
"""
function sample_model(x::Vector{Float64}, rad_sim; I::Float64=I, Δx::Float64=Δx, z_index::Int=1)
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
end
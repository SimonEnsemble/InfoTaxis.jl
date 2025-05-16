module SimulationSpace
Constants = include("Constants.jl")
RadModelStructs = include("RadModelStructs.jl")
InferenceEngine = include("InferenceEngine.jl")

using .InferenceEngine, .Constants, .RadModelStructs, LinearAlgebra, Turing, SpecialFunctions, DataFrames, StatsBase, Distributions, JLD2, Logging, LatinHypercubeSampling

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
end
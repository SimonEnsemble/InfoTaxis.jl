module InferenceEngine

Constants = include("Constants.jl")
RadModelStructs = include("RadModelStructs.jl")

using .Constants, .RadModelStructs, LinearAlgebra, Turing, SpecialFunctions, DataFrames, StatsBase, Distributions, JLD2

#############################################################################
##  ANALYTICAL (POISSON) MODEL
#############################################################################

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

	#this piece became necessary as NAN or negative values were being tested by the optimizer causing errors
	if isnan(λ) || λ < 0
		return Poisson(0.0)
	else
		return Poisson(λ)
	end
end

#############################################################################
##  TURING MCMC
#############################################################################

@model function rad_model(data; L_min::Float64=0.0, L_max::Float64=L)
	# source location
    x₀ ~ filldist(Uniform(L_min, L_max), 2)
	# source strength
	I ~ Uniform(I_min, I_max)

    for i in 1:nrow(data)
        data[i, "counts"] ~ count_Poisson(data[i, "x [m]"], x₀, I)
    end

    return nothing
end

end
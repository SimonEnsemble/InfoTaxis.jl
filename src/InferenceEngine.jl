module InferenceEngine

Constants = include("Constants.jl")
RadModelStructs = include("RadModelStructs.jl")

using .Constants, .RadModelStructs, LinearAlgebra, Turing, SpecialFunctions, DataFrames, StatsBase, Distributions, JLD2


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

#############################################################################
##  ANALYTICAL (POISSON) MODEL
#############################################################################

function attenuation_constant(x::Vector{Float64}, x₀; Σ::Float64=Σ_air)
    distance = norm(x .- x₀)
    return exp(-Σ * distance)
end


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

#############################################################################
##  TURING MCMC
#############################################################################

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

export rad_model, count_Poisson
end
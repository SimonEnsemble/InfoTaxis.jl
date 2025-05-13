module RadModelStructs

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

struct RadSim
    γ_matrix::Vector{Matrix{Float64}} #vector of matrices of gamma values. Multiply by Intensity to get counts/s, each entry of the vector represents a z-slice
    Δxy::Float64 #m, the size of each step in the x,y plane of the γ_matrix
    Δz::Float64 #m, the size of each step in the z plane
    Lxy::Float64 #m, the size of the x, y plane... assume square
    Lz::Float64 #m, the size of the z plane
	x₀::Vector{Float64} #the coordinates of the source
end

struct Environment
	env::Matrix{Int64} #unaltered mapped environment from vacuum robot
	masked_env::Matrix{Int64} #masked environment after flood fill algorith applied
	grid::Array{Union{Bool, Int64}, 3} #array of patrol grid [x coord, y coord, is_obstructed()]
	Δ::Float64 #grid spacing
end

export Obstruction, Rectangle, Circle, RadSim, Environment
end
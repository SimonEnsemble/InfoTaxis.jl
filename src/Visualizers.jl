module Visualizers
include("Constants.jl")
include("RadModelStructs.jl")

using .Constants, .RadModelStructs, CiaroMakie, ColorSchemes, DataFrames, JLD2, CSV

#############################################################################
##  EXPERIMENT SPACE VISUALIZERS
#############################################################################

"""
Visualize the robot search grid over the environment.

* `environment::Environment` - environment struct containing the matrix containing only 0's and 1's, where a 0 represents open space and a 1 represents an obstruction and the array generated from the `generate_robot_grid_matrix()` function where the first two values are the x and y coordinates of each grid location for the robot search space and the third value is the matrix of boolean values representing accessibility of locations.
* `fig_size::Int=800` - resolution control.
* `show_grid::Bool=true` - set to false to remove the grid of robot sampling points from the visual.
"""
function viz_robot_grid(
	environment; 
	fig_size::Int=800,
	show_grid::Bool=true
)
    fig = Figure(size=(fig_size, fig_size))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="rad source search space")

    heatmap!(ax, environment.masked_env; colormap=:grays)

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
		axislegend(ax, position=:lb)
	end
    return fig
end

#############################################################################
##  MODEL VISUALIZERS
#############################################################################


end
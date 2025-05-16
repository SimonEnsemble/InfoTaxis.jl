module Constants

using ColorSchemes

	# size of search space
	L  = 1000.0 # m, assuming space is square
	Δx = 10.0 # m
	
	# constant attenuation for air
	Σ_air = 0.015
	
	# Detector Parameters
	ϵ  = 0.95 #efficiency
	Δt = 1.0 #s
	A  = 0.0224 #m^2

	# source parameters
	x₀  = [250.0, 250.0]
	P_γ = 0.85 #about 85% decays emit detectable gamma
	Σ   = 0.2 #macroscopic cross section (mean free path)
	mCi = 0.050 #50 μCi
	I   = mCi * 3.7 * 10^7 * P_γ # 1mCi = 3.7*10^7 Bq
	#counts/gamma - multiply this by the value normalized to #of photons

	# colors
	colormap = reverse(vcat([ColorSchemes.hot[i] for i in 0.0:0.02:1], ColorSchemes.batlow[0.0]))

	# Locate files for which data needs to be extracted
	dir_name   = "sim_data"
	data_dir   = joinpath(dirname(@__DIR__), dir_name)
	data_files = [joinpath(data_dir, file) for file in readdir(data_dir)]

	# Turing parameters
	I_max = 5e10 #emmissions/s
    I_min = 5e4 #emmissions/s

	# robot, environment parameters
	r_velocity = 5.0 #m/s
    λ_background = 1.5 #Poisson distr lambda val for noise

    export  λ_background, r_velocity, I_min, I_max, data_files, data_dir, dir_name, colormap, L, Δx, Σ_air, ϵ, Δt, A, x₀, P_γ, Σ, mCi, I
end
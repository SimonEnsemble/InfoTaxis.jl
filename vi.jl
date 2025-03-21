### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 24a0f820-0698-11f0-1436-21b9c20c6da0
begin
	import Pkg; Pkg.activate()
	using Turing, CairoMakie, ComponentArrays, UnPack, LinearAlgebra, Bijectors
	using Turing.Variational
	using Bijectors: Scale, Shift
end

# ╔═╡ 07ec9f30-06ea-4041-b0ed-f9493a99909b
X = [
	1.0 1.0;
	1.1 1.1;
	1.4 1.3;
	1.0 1.4
]

# ╔═╡ 68542c87-8e3b-4c08-b2c1-7ccb4b0e6c70
n = size(X)[1]

# ╔═╡ 05d8ffb1-8707-4cfe-93da-ec192acc451b
d = size(X)[2]

# ╔═╡ 55828ee7-8f14-4383-90c7-efeedb2bcfcc
α = 0.4

# ╔═╡ f59b8c66-f027-426d-9cb0-11fffecea0b6
β = 1.2

# ╔═╡ edfa8523-ddeb-49d9-8450-deb6a6109661
σ = 0.01

# ╔═╡ 7b9e67a4-5477-4ec9-a14e-63cd077008a2
y = X * [α, β] .+ σ * randn(n)

# ╔═╡ 1ac292e4-8add-4441-a2cb-a70ebf216c8b
y[1]

# ╔═╡ d20e3704-078b-4574-ad77-943c35cfbddc
α * x[1, 1] + β * x[1, 2]

# ╔═╡ a254c475-4e0f-4746-b8af-f2bb8fc33efa
@model function model(x, y)
    α ~ Normal(0.0, 2.0)
    β ~ Normal(0.0, 2.0)
    for i in 1:n
        y[i] ~ Normal(α * x[i, 1] + β * x[i, 2], σ)
    end
end;

# ╔═╡ cae0d173-d762-4484-bc41-30db0d0f3f46
fm = model(X, y)

# ╔═╡ f21d55e8-5090-452d-8a28-6fbd5a4f6afd
to_constrained = inverse(bijector(fm))

# ╔═╡ 252f4ac3-68e6-4ed3-93c2-153b6ca75cb5
begin
	proto_arr = ComponentArray(; L=zeros(d, d), b=zeros(d))
	proto_axes = getaxes(proto_arr)
	num_params = length(proto_arr)

	base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))
	
	function getq(θ)
	    L, b = begin
	        @unpack L, b = ComponentArray(θ, proto_axes)
	        LowerTriangular(L), b
	    end
	    # For this to represent a covariance matrix we need to ensure that the diagonal is positive.
	    # We can enforce this by zeroing out the diagonal and then adding back the diagonal exponentiated.
	    D = Diagonal(diag(L))
	    A = L - D + exp(D) # exp for Diagonal is the same as exponentiating only the diagonal entries
	
	    b = to_constrained ∘ Shift(b) ∘ Scale(A)
	
	    return transformed(base_dist, b)
	end
end

# ╔═╡ e3b76e89-fb21-481d-95ab-726cfc58d1fc
advi = ADVI(10, 20_000)

# ╔═╡ ac5952b9-d51a-4e42-bfe7-94e461d25aea
q_full_normal = vi(
    fm, advi, getq, randn(num_params)
	# ; optimizer=Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
);

# ╔═╡ 0861875f-19af-49b0-829a-a1cb09cea647
# ╠═╡ disabled = true
#=╠═╡
A = q_full_normal.transform.inner.a
  ╠═╡ =#

# ╔═╡ 293e748b-930b-4a35-af99-cbd8995bbe51
q_full_normal

# ╔═╡ 49636c60-9823-461d-ba82-fcb2046b9a7f
C = q_full_normal.transform.inner.a

# ╔═╡ 310c5586-eb6c-4b09-ba54-0231e676b652
μ = q_full_normal.transform.outer.inner.a

# ╔═╡ d0020973-bc13-429d-8ef3-e40fc0fade91
[α, β]

# ╔═╡ 545ba451-e7a0-4e22-aa5c-a9114b727ae8
q_full_normal.transform.outer.inner.a

# ╔═╡ 57bfe47b-bdc6-4b74-bec3-211111ae11c4
posterior_samples = [rand(q_full_normal) for i = 1:100]

# ╔═╡ 92aee16a-88f0-4ad7-a617-0fc68265c36e
scatter(
	[x[1] for x in posterior_samples],
	[x[2] for x in posterior_samples]
)

# ╔═╡ Cell order:
# ╠═24a0f820-0698-11f0-1436-21b9c20c6da0
# ╠═07ec9f30-06ea-4041-b0ed-f9493a99909b
# ╠═68542c87-8e3b-4c08-b2c1-7ccb4b0e6c70
# ╠═05d8ffb1-8707-4cfe-93da-ec192acc451b
# ╠═55828ee7-8f14-4383-90c7-efeedb2bcfcc
# ╠═f59b8c66-f027-426d-9cb0-11fffecea0b6
# ╠═edfa8523-ddeb-49d9-8450-deb6a6109661
# ╠═7b9e67a4-5477-4ec9-a14e-63cd077008a2
# ╠═1ac292e4-8add-4441-a2cb-a70ebf216c8b
# ╠═d20e3704-078b-4574-ad77-943c35cfbddc
# ╠═a254c475-4e0f-4746-b8af-f2bb8fc33efa
# ╠═cae0d173-d762-4484-bc41-30db0d0f3f46
# ╠═f21d55e8-5090-452d-8a28-6fbd5a4f6afd
# ╠═252f4ac3-68e6-4ed3-93c2-153b6ca75cb5
# ╠═e3b76e89-fb21-481d-95ab-726cfc58d1fc
# ╠═ac5952b9-d51a-4e42-bfe7-94e461d25aea
# ╠═0861875f-19af-49b0-829a-a1cb09cea647
# ╠═293e748b-930b-4a35-af99-cbd8995bbe51
# ╠═49636c60-9823-461d-ba82-fcb2046b9a7f
# ╠═310c5586-eb6c-4b09-ba54-0231e676b652
# ╠═d0020973-bc13-429d-8ef3-e40fc0fade91
# ╠═545ba451-e7a0-4e22-aa5c-a9114b727ae8
# ╠═57bfe47b-bdc6-4b74-bec3-211111ae11c4
# ╠═92aee16a-88f0-4ad7-a617-0fc68265c36e

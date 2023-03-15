# Functions to aid in discretization 

"""
	grid_generator

Create a grid generator from lower and upper point extremeties with spacing in delta
"""
function grid_generator(L, U, δ)
	generator = nothing
	dim_generators = []
    δ_tight = zeros(size(δ))

    # δ is just the desired discretization - to make it work with the L and U bounds, we can adjust it slightly.
	for i=1:length(L)
        N_tight = Int(floor((U[i] - L[i])/δ[i]))
        δ_tight[i] = (U[i] - L[i])/N_tight
        gen = (L[i]:δ_tight[i]:U[i])
        push!(dim_generators, gen[1:end-1])
	end

	generator = Iterators.product(dim_generators...)
	return generator, δ_tight
end

function vertices_to_extents(X)
    ndims = size(X,1)
    extents = zeros(ndims,2)
    for i=1:ndims
        extents[i,:] = [minimum(X[i,:]), maximum(X[i,:])]
    end
    return extents
end

function SA_to_polytope()
    nothing
end

function SA_to_extent(state)
    return [state[:,1] state[:,end-1]]
end

function extent_to_SA(extent)
	ndims = length(extent)
	ncols = 2^ndims # Probably always 2
	res = zeros(ndims, ncols)

    for i in 0:ncols-1
        for j in 0:ndims-1
            res[j+1, i+1] = (i >> j) & 1 == 0 ? extent[j+1][1] : extent[j+1][2]
        end
    end

    # Tmp fix for ordering
    if ndims > 1
        temp_col = copy(res[:,end-1])
        res[:, end-1] = res[:, end]
        res[:, end] = temp_col
    end
    return SMatrix{ndims, ncols}(res)
end

function lower_to_extent(L, δ)
	extent = [[L[i], L[i] + δ[i]] for i=1:length(L)]
	return extent 
end

function lower_to_SA(L, δ)
    extent = lower_to_extent(L, δ)
    return extent_to_SA(extent)
end
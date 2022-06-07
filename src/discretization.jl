# Functions to aid in discretization 

"""
	grid_generator

Create a grid generator from lower and upper point extremeties with spacing in delta
"""
function grid_generator(L, U, δ)
	generator = nothing
	dim_generators = []
	for i=1:length(L)
		push!(dim_generators, (L[i]:δ[i]:U[i])[1:end-1])
	end

	generator = Iterators.product(dim_generators...)
	return generator
end

function vertices_to_extents(X)
    ndims = size(X,1)
    extents = zeros(ndims,2)
    for i=1:ndims
        extents[i,:] = [minimum(X[i,:]), maximum(X[i,:])]
    end
    return extents
end

" Discretize a given extent into smaller extents with a grid size of delta.
# Arguments
- `set::Dict` - Set to discretize
- `grid_sizes::Dict` - Discretization delta for each component
"
function discretize(L, U, grid_spacing)
    grid = CartesianGrid(L, U, grid_spacing)
    return grid 
end

function polytope_to_SA(X)
    b = @view(X.vertices[1:end])
    return hcat([b[i].coords for i=1:length(b)]...)
end

function SA_to_polytope()
    nothing
end

function extent_to_SA(extent)
	ndims = length(extent)
	ncols = 2^ndims # Probably always 2
	res = zeros(ndims, ncols)

    # TODO: Extend to any dim?
    if ndims == 2
        res[:,1] = [extent[1][1], extent[2][1]]
	    res[:,2] = [extent[1][2], extent[2][1]]
	    res[:,3] = [extent[1][2], extent[2][2]]
	    res[:,4] = [extent[1][1], extent[2][2]]
    elseif ndims == 3 
        zdims = ndims - 1
        zoffset = 0
        for i=1:zdims
            res[:,1+zoffset] = [extent[1][1], extent[2][1], extent[3][i]]
            res[:,2+zoffset] = [extent[1][2], extent[2][1], extent[3][i]]
            res[:,3+zoffset] = [extent[1][2], extent[2][2], extent[3][i]]
            res[:,4+zoffset] = [extent[1][1], extent[2][2], extent[3][i]]
            zoffset += 4
        end
    elseif ndims == 4
        zdims = ndims - 1
        zoffset = 0
        for j=1:2
            for i=1:2
                res[:,1+zoffset] = [extent[1][1], extent[2][1], extent[3][i], extent[4][j]]
                res[:,2+zoffset] = [extent[1][2], extent[2][1], extent[3][i], extent[4][j]]
                res[:,3+zoffset] = [extent[1][2], extent[2][2], extent[3][i], extent[4][j]]
                res[:,4+zoffset] = [extent[1][1], extent[2][2], extent[3][i], extent[4][j]]
                zoffset += 4
            end 
        end
    else
        @error("Invalid number of dimensions!")
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
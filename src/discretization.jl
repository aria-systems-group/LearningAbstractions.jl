
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
    elseif ndims > 2
        zdims = ndims - 1
        zoffset = 0
        for i=1:zdims
            res[:,1+zoffset] = [extent[1][1], extent[2][1], extent[3][i]]
            res[:,2+zoffset] = [extent[1][2], extent[2][1], extent[3][i]]
            res[:,3+zoffset] = [extent[1][2], extent[2][2], extent[3][i]]
            res[:,4+zoffset] = [extent[1][1], extent[2][2], extent[3][i]]
            zoffset += 4
        end
    end

	return SMatrix{ndims, ncols}(res)
end
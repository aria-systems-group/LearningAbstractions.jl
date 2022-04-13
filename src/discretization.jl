
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
    b = @view(X.vertices[1:4])
    return hcat([b[i].coords for i=1:length(b)]...)
end

function SA_to_polytope()
    nothing
end
# Tools for visualizing abstractions

"""
    vertices_to_shape

Converts the indeces in a static array into a shape object for plotting.
"""
function vertices_to_shape(X; idx=(1,2))
    return Plots.Shape(Vector(X[idx[1],:]), Vector(X[idx[2],:])) 
end
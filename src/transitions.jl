"""
    distance

Calculate the minimum distance between two convex sets. If 0., the sets intersect.
"""
function distance(X::Polytope, Y::Polytope)
    X_SA = polytope_to_SA(X)
    Y_SA = polytope_to_SA(Y)
    return distance(X_SA, Y_SA)
end

function distance(X::SMatrix, Y::SMatrix)
    n = size(X,1)
    dir = @SVector(rand(n)) .- 0.5
    # TODO: Can this be made faster?
    return minimum_distance(X, Y, dir)
end

function generate_all_transitions(grid, images, all_state_means, all_image_means, full_set; gp_rkhs_info=nothing, σ_bounds_all=nothing)
    num_states = length(grid) + 1 # All states plus the unsafe state!
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 
    P̌[1:end-1, 1:end-1], P̂[1:end-1, 1:end-1] = generate_pairwise_transitions(grid, images, all_state_means, all_image_means, gp_rkhs_info=gp_rkhs_info, σ_bounds_all=σ_bounds_all) 

    for i in 1:num_states-1 
        σ_bounds = isnothing(gp_rkhs_info) ? nothing : σ_bounds_all[i]  
        p̌, p̂ = transition_inverval(images[i], full_set, gp_rkhs_info=gp_rkhs_info, σ_bounds=σ_bounds) 
        P̌[i,end] = 1 - p̂
        P̂[i,end] = 1 - p̌ 
    end 
    P̌[end,end] = 1.
    P̂[end,end] = 1.
    
    return P̌, P̂ 
end

function generate_pairwise_transitions(grid, images, all_state_means, all_image_means; gp_rkhs_info=nothing, σ_bounds_all=nothing)

    num_states = length(grid)
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 

    p = Progress(num_states^2, desc="Computing transition intervals...", dt=0.01)
    for i in 1:num_states 
        image = images[i]
        mean_image = all_image_means[i]
        σ_bounds = isnothing(gp_rkhs_info) ? nothing : σ_bounds_all[i]  

        for j in 1:num_states 
            statep_sa = grid[j]
            if true || fast_check(mean_image, all_state_means[j], 0.0, 0.125, 0.125) #! FIX THIS
                P̌[i,j], P̂[i,j] = transition_inverval(image, statep_sa, gp_rkhs_info=gp_rkhs_info, σ_bounds = σ_bounds) 
            end
            next!(p)
        end
    end
    return P̌, P̂ 
end

"Determine the set of states that will have non-zero transition probability upper bounds."
function fast_check(mean_pt, mean_target, ϵ_crit, image_radius, set_radius)
    flag = norm(mean_pt - mean_target) < ϵ_crit + image_radius + set_radius
    return flag
end

function transition_inverval(X,Y; gp_rkhs_info=nothing, σ_bounds=nothing)
    dis = distance(X, Y)
    #===
    Full or Partial Intersection
    ===#
    if (dis <= 0.)
        p̂ = 1.              # UB result for full + partial intersection
        containment_flag, min_distance = containment_check(X, Y)
        #===
        Full Intersection
        ===#
        if containment_flag     
            if !isnothing(gp_rkhs_info)
                p̌ = chowdhury_rkhs_prob_vector(gp_rkhs_info, σ_bounds, min_distance) 
            else
                p̌ = 1.          
            end
        else
            p̌ = 0           # LB result for partial intersection
        end
    #===
    No Intersection
    ===#
    else
        if !isnothing(gp_rkhs_info)
            p̂ = 1 - chowdhury_rkhs_prob_vector(gp_rkhs_info, σ_bounds, dis)
        else
            p̂ = 0 
        end
        p̌ = 0
    end
    return [p̌, p̂]
end

function containment_check(shape1,shape2, axes=nothing)
    int_result = false 
    min_distance = Inf
    n_dims = size(shape1,1)
    axes = I+zeros(n_dims, n_dims)

    if isnothing(axes)
        axes = get_axes([shape1, shape2])
    end

    num_axes = size(axes)[2]
    for i=1:num_axes
        axis = axes[:, i:i]
        p1 = project(shape1, axis)
        p2 = project(shape2, axis)
        # overlap = (p2[1] <= p1[1] <= p2[2] || p2[1] <= p1[2] <= p2[2] ||
        #            p1[1] <= p2[1] <= p1[2] || p1[1] <= p2[2] <= p1[2] )
        contain = p2[1] <= p1[1] <= p2[2] && p2[1] <= p1[2] <= p2[2]
        if contain
            min_distance = minimum([abs(p1[1] - p2[1]), abs(p1[2] - p2[2]), min_distance])
            int_result = true 
        else
            int_result = false
            break
        end
    end
    return int_result, min_distance

end

" Projects the shape vertices onto the axis. 
# Arguments
- `shape::Dict` - Shape with entry `vertices`
- `axis` - Axis of projection
"
function project(shape, axis)
    dots = [(vertex'*axis)[1] for vertex in eachcol(shape)]
    min = minimum(dots)
    max = maximum(dots)
    return [min, max]
end

" Gets the principal normals of each shape.
# Arguments
- `shapes::Vector` - Vector with shape definitions 
"
function get_axes(shapes::Vector)
    axes = []
    for shape in shapes
        # num_v = length(shape["vertices"])
        num_v = size(shape,2)
        for i=1:num_v
            normals = nullspace(Array(shape[:,i])-(i==num_v ? Array(shape[:,1]) : Array(shape[:,i+1]))) 
            if isempty(axes)
                axes = normals
            else
                # TODO: remove redundant axes
                axes = hcat([axes normals])
            end 
        end
    end
    return axes
end

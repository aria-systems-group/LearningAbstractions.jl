"""
    distance

Calculate the minimum distance between two convex sets. If 0., the sets intersect.
"""
function distance(X::Polytope, Y::Polytope)
    X_SA = polytope_to_SA(X)
    Y_SA = polytope_to_SA(Y)
    return distance(X_SA, Y_SA)
end

using EnhancedGJK
import CoordinateTransformations: IdentityTransformation, Translation 

function distance(X::SMatrix, Y::SMatrix)
    n = size(X,1)
    dir = @SVector(rand(n)) .- 0.5
    # TODO: Can this be made faster?

    if n == 4
        # This is a hacky way to get the distance
        s1 = SVector{16}(eachcol(X))
        s2 = SVector{16}(eachcol(Y))

        cache = CollisionCache(s1, s2)
        result = gjk!(cache, IdentityTransformation(), IdentityTransformation())
        if result.in_collision
            return 0.
        else
            return separation_distance(result)
        end
    end
    return minimum_distance(X, Y, dir, atol=1e-6)
end

function generate_all_transitions(grid, images, full_set; gp_rkhs_info=nothing, σ_bounds_all=nothing, ϵ_manual=nothing)
    num_states = length(grid) + 1 # All states plus the unsafe state!
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 
    P̌[1:end-1, 1:end-1], P̂[1:end-1, 1:end-1] = generate_pairwise_transitions(grid, images, gp_rkhs_info=gp_rkhs_info, σ_bounds_all=σ_bounds_all, ϵ_manual=ϵ_manual) 
    @info "Finished generating pairwise transitions."
    for i in 1:num_states-1 
        σ_bounds = isnothing(gp_rkhs_info) ? nothing : σ_bounds_all[i]  
        p̌, p̂ = transition_inverval(images[i], full_set, gp_rkhs_info=gp_rkhs_info, σ_bounds=σ_bounds, ϵ_manual=ϵ_manual) 
        P̌[i,end] = 1 - p̂
        P̂[i,end] = 1 - p̌ 
    end 
    @info "Finished generating transitions to unsafe state."
    P̌[end,end] = 1.
    P̂[end,end] = 1.
    
    return P̌, P̂ 
end

function generate_pairwise_transitions(states, images; gp_rkhs_info=nothing, σ_bounds_all=nothing, ϵ_manual=nothing)

    num_states = length(states)
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 

    # calculate state and image means for fast check
    all_state_means = state_means(states)
    all_image_means = state_means(images)

    # TODO: Replace this when grid is replaced
    state_radius = norm(states[1][:,1] - states[1][:,end-1])/2
    fast_checks = 0

    p = Progress(num_states^2, desc="Computing transition intervals...", dt=30)

    for i in 1:num_states 
        image = images[i]
        image_radius = norm(image[:,1] - image[:,end-1])/2
        mean_image = all_image_means[i]
        if !isnothing(gp_rkhs_info)
            σ_bounds = σ_bounds_all[i]
            # ! TODO: Generalize
            RKHS_bound_local = gp_rkhs_info.f_sup / sqrt(exp(-1/2*(2*state_radius)^2/0.6))
            ϵ_crit = calculate_ϵ_crit(gp_rkhs_info, σ_bounds, local_RKHS_bound=RKHS_bound_local)
        else
            σ_bounds = nothing
            RKHS_bound_local = nothing
            ϵ_crit = 0. 
        end

        for j in 1:num_states 
            statep_sa = states[j]
            if fast_check(mean_image, all_state_means[j], ϵ_crit, image_radius, state_radius) 
                P̌[j,i], P̂[j,i] = transition_inverval(image, statep_sa, gp_rkhs_info=gp_rkhs_info, σ_bounds=σ_bounds, ϵ_manual=ϵ_manual, local_RKHS_bound=RKHS_bound_local) 
            else
                fast_checks += 1
            end
            next!(p)
        end
    end

    num_trans = num_states^2
    fast_frac = fast_checks/num_trans
    @info "$fast_checks / $num_trans ($fast_frac) states passed quick check."
    # Take the transpose to get the correct 
    return P̌', P̂' 
end

"Determine the set of states that will have non-zero transition probability upper bounds."
# ! Need to verify this fast check
function fast_check(mean_pt, mean_target, ϵ_crit, image_radius, set_radius)
    flag = norm(mean_pt - mean_target) < ϵ_crit + image_radius + set_radius
    return flag
end

function transition_inverval(X,Y; gp_rkhs_info=nothing, σ_bounds=nothing, ϵ_manual=nothing, local_RKHS_bound=nothing)
    dis = distance(X, Y)
    #===
    Full or Partial Intersection
    ===#
    if (dis <= 1e-4)
        p̂ = 1.              # UB result for full + partial intersection
        containment_flag, min_distance = containment_check(X, Y)
        #===
        Full Intersection
        ===#
        if containment_flag     
            if !isnothing(gp_rkhs_info)
                p̌ = chowdhury_rkhs_prob_vector(gp_rkhs_info, σ_bounds, min_distance,local_RKHS_bound=local_RKHS_bound) 
            else
                p̌ = 1.          
            end
        else
            p̌ = 0.           # LB result for partial intersection
        end
    #===
    No Intersection
    ===#
    else
        if !isnothing(gp_rkhs_info)
            p̂ = 1 - chowdhury_rkhs_prob_vector(gp_rkhs_info, σ_bounds, dis, local_RKHS_bound=local_RKHS_bound)
        else
            p̂ = 0. 
        end
        p̌ = 0.
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

"""
    calculate_ϵ_crit

Calculates the value of epsilon that a.s. bounds the RKHS regression error.
"""
function calculate_ϵ_crit(gp_info, σ_bounds; local_RKHS_bound=nothing)
    ϵ = 0.01
    P = 0.0
    while P != 1.0
        P = chowdhury_rkhs_prob_vector(gp_info, σ_bounds, ϵ, local_RKHS_bound=local_RKHS_bound) 
        ϵ *= 1.5 
    end
    return ϵ
end


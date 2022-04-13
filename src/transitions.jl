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

function generate_all_transitions(grid, images, all_state_means, all_image_means, full_set)
    num_states = length(grid) + 1 # All states plus the unsafe state!
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 
    P̌[1:end-1, 1:end-1], P̂[1:end-1, 1:end-1] = generate_pairwise_transitions(grid, images, all_state_means, all_image_means) 

    for i in 1:num_states-1 
        image = images[i]
        p̌, p̂ = transition_invervals(image, full_set) 
        P̌[i,end] = 1 - p̂
        P̂[i,end] = 1 - p̌ 
    end 
    P̌[end,:] .= 1.
    P̂[end,:] .= 1.
    
    return P̌, P̂ 
end

function generate_pairwise_transitions(grid, images, all_state_means, all_image_means)

    num_states = length(grid)
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 

    for i in 1:num_states 
        image = images[i]
        mean_image = all_image_means[i]

        for j in 1:num_states 
            statep_sa = grid[j]
            if fast_check(mean_image, all_state_means[j], 0., 0.25, 0.25) 
                P̌[i,j], P̂[i,j] = transition_invervals(image, statep_sa) 
            end
        end
    end
    return P̌, P̂ 
end

"Determine the set of states that will have non-zero transition probability upper bounds."
function fast_check(mean_pt, mean_target, ϵ_crit, image_radius, set_radius)
    flag = norm(mean_pt - mean_target) < ϵ_crit + image_radius + set_radius
    return flag
end

function transition_invervals(X,Y;)
    dis = distance(X, Y)
    if (dis <= 0)
        p̂ = 1.
        # TODO: Make this better
        b1 = Meshes.Box(Point(X[:,1]), Point(X[:,3]))
        b2 = Meshes.Box(Point(Y[:,1]), Point(Y[:,3]))
        if b1 ⊆ b2
            p̌ = 1
        else
            p̌ = 0
        end
    else
        p̂ = 0.
        p̌ = 0
    end
    return [p̌, p̂]
end

function containment_check(X,Y)

    # For now, assume same # of vertices 

end
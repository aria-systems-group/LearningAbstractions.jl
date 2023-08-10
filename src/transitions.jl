"""
    intersects

Check if two hyperrectangles intersect (not w/ non-zero measure)
"""
function intersects(X, Y)
    # TODO: Generalize for vector input
    res = true
    # project onto each axis - hyperrectangle only
    for i in axes(X,1)
         xp_1 = (Y[i,1] ≤ X[i,1] ≤ Y[i,2]) || (Y[i,1] ≤ X[i,2] ≤ Y[i,2]) 
         xp_2 = (X[i,1] ≤ Y[i,1] ≤ X[i,2]) || (X[i,1] ≤ Y[i,2] ≤ X[i,2])  
         res = res && (xp_1 || xp_2)
    end

    return res 
end


function generate_all_transitions(states, images, full_set; process_noise_dist=nothing, gp_rkhs_info=nothing, σ_bounds_all=nothing, ϵ_manual=nothing, η_manual=nothing, local_gp_metadata=nothing, P̌_hot=nothing, P̂_hot=nothing, target_idxs_dict=nothing, multibounds_flag=false)
    num_states = length(states) + 1 # All states plus the unsafe state!
    P̌ = spzeros(num_states, num_states)
    P̂ = spzeros(num_states, num_states) 

    P̌[1:end-1, 1:end-1], P̂[1:end-1, 1:end-1] = generate_pairwise_transitions(states, images, process_noise_dist=process_noise_dist, gp_rkhs_info=gp_rkhs_info, σ_bounds_all=σ_bounds_all, ϵ_manual=ϵ_manual, η_manual=η_manual, local_gp_metadata=local_gp_metadata, target_idxs_dict=target_idxs_dict, multibounds_flag=multibounds_flag) 

    hot_idx = []
    if !isnothing(P̌_hot) && !isnothing(P̂_hot)
        num_hot = size(P̌_hot)[1]
        hot_idx = 1:num_hot-1

        nh2 = num_hot^2
        ns2 = num_states^2
        fr = nh2/ns2
        @info "Reusing $nh2/$ns2 ($fr) transitions"

        P̌[1:num_hot-1, 1:num_hot-1] = P̌_hot[1:end-1, 1:end-1]
        P̂[1:num_hot-1, 1:num_hot-1] = P̂_hot[1:end-1, 1:end-1]
        P̌[1:num_hot-1, end] = P̌_hot[1:end-1, end]
        P̂[1:num_hot-1, end] = P̂_hot[1:end-1, end]
    end
    @info "Finished generating pairwise transitions."
    p_vec = zeros(size(images[1],1))
    for i in setdiff(1:num_states-1, hot_idx)   # Always calculating transitions to the unsafe set!
        σ_bounds = isnothing(gp_rkhs_info) ? nothing : σ_bounds_all[i]  
        p̌, p̂ = transition_inverval(images[i], full_set, p_vec, process_noise_dist=process_noise_dist, gp_rkhs_info=gp_rkhs_info, σ_bounds=σ_bounds, ϵ_manual=ϵ_manual, η_manual=η_manual, local_gp_metadata=local_gp_metadata, multibounds_flag=multibounds_flag) 
        P̌[i,end] = 1 - p̂
        P̂[i,end] = 1 - p̌ 
    end 

    @info "Finished generating transitions to unsafe state."
    P̌[end,end] = 1.
    P̂[end,end] = 1.
    
    return P̌, P̂ 
end

function generate_pairwise_transitions(states, images; process_noise_dist=nothing, gp_rkhs_info=nothing, σ_bounds_all=nothing, ϵ_manual=nothing, η_manual=nothing, local_gp_metadata=nothing, target_idxs_dict=nothing, multibounds_flag=false)

    dims = size(images[1],1)
    num_states = length(states)

    # calculate state and image means for fast check
    all_state_means = state_means(states)
    all_image_means = state_means(images)

    fast_checks = 0
    skipped_idxs = 0
    checked_idxs = 0

    p = Progress(num_states^2, desc="Computing transition intervals...", dt=status_bar_period)
    η_crit = !isnothing(process_noise_dist) ? calculate_η_crit(process_noise_dist) : 0.0

    nthreads = Threads.nthreads()
    # Allocate p_rkhs vectors
    p_rkhs_vec_all = [zeros(dims) for _ in 1:nthreads]
    # Allocate all temp P̌, P̂ matrices
    P̌_temp = [spzeros(num_states, num_states) for _ in 1:nthreads]
    P̂_temp = [spzeros(num_states, num_states) for _ in 1:nthreads]

    for i in 1:num_states 
        image = images[i]
        image_radius = norm(image[:,1] - image[:,end-1])/2
        mean_image = all_image_means[i]
        state_radius_i = norm(states[i][1:dims,1] - states[i][1:dims,end-1])/2 

        if !isnothing(gp_rkhs_info)
            σ_bounds = σ_bounds_all[i]
            # TODO: Generalize the GP kernel
            RKHS_bound_local = gp_rkhs_info.f_sup / sqrt(exp(-1/2*(2*state_radius_i)^2/exp(0.65)))
            ϵ_crit = calculate_ϵ_crit(gp_rkhs_info, σ_bounds, local_RKHS_bound=RKHS_bound_local)
        else
            σ_bounds = nothing
            RKHS_bound_local = nothing
            ϵ_crit = 0. 
        end

        if !isnothing(target_idxs_dict) && i∈keys(target_idxs_dict)
            idxs_to_check = target_idxs_dict[i]
            l = length(idxs_to_check)
            skipped_idxs += num_states - l
            checked_idxs += l
        else
            idxs_to_check = 1:num_states 
            checked_idxs += num_states
        end

        # Threads.@threads 
        for j in idxs_to_check
            statep_sa = states[j]
            state_radius_j = norm(statep_sa[1:dims,1] - statep_sa[1:dims,end-1])/2 
            if fast_transitions_check && fast_check(mean_image, all_state_means[j][1:dims], ϵ_crit, η_crit, image_radius, state_radius_j) 
                res = transition_inverval(image, statep_sa, p_rkhs_vec_all[Threads.threadid()], process_noise_dist=process_noise_dist, gp_rkhs_info=gp_rkhs_info, σ_bounds=σ_bounds, ϵ_manual=ϵ_manual, η_manual=η_manual, local_RKHS_bound=RKHS_bound_local, local_gp_metadata=local_gp_metadata, multibounds_flag=multibounds_flag) # ! seg fault here
                P̌_temp[Threads.threadid()][j,i] = res[1]
                P̂_temp[Threads.threadid()][j,i] = res[2]
            else
                fast_checks += 1
            end
            next!(p)
        end
    end

    P̌ = sum(P̌_temp)
    P̂ = sum(P̂_temp)

    num_trans = num_states^2
    skipped_frac = skipped_idxs / num_trans
    @info "$skipped_idxs / $num_trans ($skipped_frac) transition pairs skipped."
    fast_frac = fast_checks/checked_idxs
    @info "$fast_checks / $checked_idxs ($fast_frac) transition pairs passed quick check."
    # Take the transpose to get the correct 
    return P̌', P̂' 
end

"Determine the set of states that will have non-zero transition probability upper bounds."
function fast_check(mean_pt, mean_target, ϵ_crit, η_crit, image_radius, set_radius)
    flag = norm(mean_pt - mean_target) < ϵ_crit + η_crit + image_radius + set_radius
    return flag
end

function transition_inverval(X, Y, p_rkhs; process_noise_dist=nothing, gp_rkhs_info=nothing, σ_bounds=nothing, ϵ_manual=nothing, local_RKHS_bound=nothing, local_gp_metadata=nothing, multibounds_flag=false, η_manual=nothing)
    # Get the dimensions of the states - if control is embedded, modify to only look at state-space
    dims = size(X,1) 
    if dims < size(Y,1) # todo: holdover from control as state
        Yr = Y[1:dims, 1:dims^2] # TODO: This is erroring out for 1D case
        Y = SMatrix{dims, 2^dims}(Yr)
    end

    intersect_flag = intersects(SA_to_extent(X), SA_to_extent(Y))

    if intersect_flag
        containment_flag, _ = containment_check(X, Y)
    else
        containment_flag = false
    end
    
    Pr_process = ones(dims)     # todo: we can reuse these vectors
    Pr_learning = ones(dims)

    dis_comps = zeros(dims, 2)
    dis_fcn!(dis_comps, X, Y)
    @info dis_comps

    η_offset = 0
    if !isnothing(process_noise_dist) && !(intersect_flag && !containment_flag)
        if !isnothing(η_manual)       # todo: kinda janky
            η_offset = η_manual
        else
            # if !isnothing(ϵ_crit)
            #     η_offset = minimum(dis_comps[:,1]) - ϵ_crit
            #     if η_offset < 0
            #         η_offset = 0.15
            #     end
            # else
            # TODO: this is incorrect, need to compensate for $η$ then calculate the cdf. need to threshold if zero
            η_offset = 0.05 

            # end
        end

        # this is still incorrect for the min case...
        mine = min(η_offset, minimum(dis_comps[:,1])) # use zero or eta_offset essentially
        Pr_process[:] .= sqrt(abs_cdf(process_noise_dist, mine)) # Per dimension, assuming the same on each axis
        # Pr_process[:] .= 1.0 - cdf(process_noise_dist, -η_offset)
    end
    offset_distance!(dis_comps, η_offset)

    if !isnothing(gp_rkhs_info) && !(intersect_flag && !containment_flag)
        Pr_learning[:] = rkhs_prob_vector(gp_rkhs_info, σ_bounds, dis_comps[:,1], local_RKHS_bound=local_RKHS_bound, local_gp_metadata=local_gp_metadata, p_rkhs=p_rkhs)
    end
    # maxe = maximum(dis_comps[:,2])
    # @info Pr_process 
    # Pr_process[:] .=  sqrt(abs_cdf(process_noise_dist, maxe)) 
    p̂ = upper_bound_prob(intersect_flag, Pr_process, Pr_learning)
    # Pr_process[:] .= cdf(process_noise_dist, 1.95) - cdf(process_noise_dist, η_offset) 
    # Pr_process[:] .=  sqrt(abs_cdf(process_noise_dist, mine)) 
    p̌ = lower_bound_prob(containment_flag, Pr_process, Pr_learning)

    @assert p̌ <= p̂
    return [p̌, p̂]
end

"""
    lower_bound_prob

Computes the lower-bound probability of transition between states given the distance components between the states.
"""
function lower_bound_prob(containment_flag::Bool, constraint1_probs::Vector{Float64}, constraint2_probs::Vector{Float64})
    if !containment_flag # some of the image lies outside the target set
        return 0
    end
    p̌ = prod(constraint1_probs.*constraint2_probs) 
    return p̌
end

"""
    upper_bound_prob

Computes the lower-bound probability of transition between states given the distance components between the states.
"""
function upper_bound_prob(intersect_flag::Bool, constraint1_probs::Vector{Float64}, constraint2_probs::Vector{Float64})
    if intersect_flag
        return 1 
    end
    p̂ = prod(1 .- constraint1_probs.*constraint2_probs) 
    return p̂
end

function offset_distance!(distances, offset)
    distances[:,1] .-= offset
    [distances[i,1] = distances[i,1] < 0.0 ? 0.0 : distances[i,1] for i in axes(distances,1)]
    distances[:,2] .+= offset
end

function dis_fcn!(res, X::AbstractMatrix, Y::AbstractMatrix)
    for i in axes(X,1)
        res[i,1] = minimum(hcat([abs.(unique(X[i,:]) .- yu) for yu in unique(Y[i,:])]...))
        res[i,2] = maximum(hcat([abs.(unique(X[i,:]) .- yu) for yu in unique(Y[i,:])]...))
    end
end

function dis_fcn!(res, X::SVector, Y::SMatrix)
    for i in eachindex(X)
        res[i,1] = minimum(hcat([abs.(X[i] .- yu) for yu in unique(Y[i,:])]...))
        res[i,2] = maximum(hcat([abs.(X[i] .- yu) for yu in unique(Y[i,:])]...))    
    end
end

function abs_cdf(pdf, val)
    return cdf(pdf, val) - cdf(pdf, -val)
end

function abs_cdf(pdf, val_lb, val_ub)
    p_ub = abs_cdf(pdf, val_ub)
    p_lb = abs_cdf(pdf, val_lb)
    return p_ub - p_lb
end

function containment_check(shape1::AbstractMatrix,shape2::AbstractMatrix, axes=nothing)
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

function containment_check(x::SVector, Y::SMatrix; axes=nothing)
    min_distance = Inf
    xinq = [Y[i,1]<=x[i]<= Y[i,end-1] for i in eachindex(x)]
    int_result = sum(xinq) == length(x)
    if int_result
        min_distance = minimum([min(abs(Y[i,1]-x[i]), abs(Y[i,end-1]-x[i])) for i in eachindex(x)])
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
    ϵ = 0.001
    P = 0.0
    while P != 1.0
        P = rkhs_prob_vector_single(gp_info, σ_bounds, ϵ, local_RKHS_bound=local_RKHS_bound) 
        ϵ *= 1.5 
    end
    return ϵ
end

""" 
    calculate_η_crit

Calculates the value of η that a.s. bounds the process noise
"""
function calculate_η_crit(dist)
    η = 0.01
    P = 0.0
    while P != 1.0
        P = cdf(dist, η)
        η *= 2
    end
    return η
end


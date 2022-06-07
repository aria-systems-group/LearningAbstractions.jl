"""
    uniform_refinement

Refine a discrete state uniformly and update the region info dict.
"""
function uniform_refinement(state)
    deltas = (state[:,end-1] - state[:,1])/2
    refined_grid = grid_generator(state[:,1], state[:,end-1], deltas)
    new_states = [lower_to_SA(gl, deltas) for gl in refined_grid]
    return new_states
end

"""
    refine_abstraction

Refine the abstraction.
"""
# TODO: There is a lot of redundant code here with the find images function. Consolidate when I have more time. 
function refine_abstraction(config_filename, all_states_SA, all_state_images, all_σ_bounds, states_to_refine; refinement_dirname=nothing)
    f = open(config_filename)
	config = TOML.parse(f)
	close(f)
	
    results_dir = config["results_directory"]
	# state_filename = "$results_dir/states.bson"
	# imdp_filename = "$results_dir/imdp.bson"
	gps_filename = "$results_dir/gps.bson"

    L = SA_F64[config["workspace"]["lower"]...]
	U = SA_F64[config["workspace"]["upper"]...]
	X_extent = [[l u] for (l, u) in zip(L,U)]

    refinement_dir = isnothing(refinement_dirname) ? "$results_dir/refined" : "$results_dir/$refinement_dirname"
    !isdir(refinement_dir) && mkpath(refinement_dir)

    # Load the GPs
    res = BSON.load(gps_filename)
    gps = res[:gps]
    neg_gps = []
	for gp in gps
		neg_gp = deepcopy(gp)
		neg_gp.alpha *= -1
		push!(neg_gps, neg_gp)
	end
    gp_info = res[:info]

    # Generate new discretization
    new_states_list = []
    for state_to_refine in states_to_refine
        new_states = uniform_refinement(all_states_SA[state_to_refine])
        push!(new_states_list, new_states...)
    end

    # delete the old states
    all_states_refined = copy(all_states_SA)
    all_state_images_refined = copy(all_state_images)
    all_σ_bounds_refined = copy(all_σ_bounds)

    deleteat!(all_states_refined, states_to_refine)
    push!(all_states_refined, new_states_list...)

    # delete old image bounds
    deleteat!(all_state_images_refined, states_to_refine)
    deleteat!(all_σ_bounds_refined, states_to_refine)

    # Generate new posterior bounds
    p = Progress(length(new_states_list), desc="Computing image bounds...", dt=0.01)
    Threads.@threads for new_state in new_states_list
        image, σ_bounds = LearningAbstractions.GPBounding.bound_image([new_state[:,1], new_state[:,end-1]], gps, neg_gps)
        push!(all_state_images_refined, extent_to_SA(image))
        push!(all_σ_bounds_refined, σ_bounds)
        next!(p)
    end

    all_state_means = [0.5*(s[:,1] + s[:,end-1]) for s in all_states_refined]
	all_image_means = [0.5*(s[:,1] + s[:,end-1]) for s in all_state_images_refined] 

    P̌, P̂ = LearningAbstractions.generate_all_transitions(all_states_refined, all_state_images_refined, all_state_means, all_image_means, LearningAbstractions.extent_to_SA(X_extent), gp_rkhs_info=gp_info, σ_bounds_all=all_σ_bounds_refined)

    @info "Saving abstraction info to $refinement_dir"
    state_refined_filename = "$refinement_dir/states.bson"
	imdp_refined_filename = "$refinement_dir/imdp.bson"

    bson(state_refined_filename, Dict(:states => all_states_refined,
                        :images => all_state_images_refined,
                        :bounds => all_σ_bounds_refined,
                        :state_means => all_state_means,
                        :image_means => all_image_means)
    )

    bson(imdp_refined_filename, Dict(:Pcheck => P̌, :Phat => P̂))
    return P̌, P̂, all_states_refined, all_state_means, refinement_dir, all_state_images_refined, all_σ_bounds_refined
end
# Functions for loading, saving, reloading, etc.

function load_states(states_filename::String)
    state_dict = BSON.load(states_filename)
    return state_dict[:states], state_dict[:images], state_dict[:bounds]
end

function load_imdp(imdp_filename::String)
    imdp_dict = BSON.load(imdp_filename)
    P̌ = imdp_dict[:Pcheck]
    P̂ = imdp_dict[:Phat]
    return P̌, P̂ 
end

function load_results(state_filename, imdp_filename; reuse_states=true, reuse_results=true)
    if (reuse_states || reuse_results) && isfile(state_filename)
		@info "Loading existing states from $state_filename"
		all_states_SA,
		all_state_images,
		all_state_σ_bounds = load_states(state_filename)
		
		if reuse_results && isfile(imdp_filename)
            @info "Loading existing IMDP info from $imdp_filename"
            P̌, P̂ = load_imdp(imdp_filename)
			# > If we get here, refinement is exactly the same!
        else 
            P̌ = nothing 
            P̂ = nothing
		end
    else
        all_states_SA = nothing
        all_state_images = nothing 
        all_state_σ_bounds = nothing
        P̌ = nothing
        P̂ = nothing
	end

    return all_states_SA, all_state_images, all_state_σ_bounds, P̌, P̂
end
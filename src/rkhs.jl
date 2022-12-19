# Functions related to the RKHS stuff

struct GPRelatedInformation  # One for each set of GPs
	γ_bounds	
	RKHS_norm_bounds
	logNoise	
	post_scale_factors
	Kinv
	f_sup	
	measurement_noise
	process_noise
end

"""
Create the GP Info structure componenets.
"""
function create_gp_info(gps, σ_noise, diameter_domain, sup_f; process_noise=false)
	γ_bds = []
	RKHS_norm_bounds = []
	logNoise = []
	post_scale_factors = []
	for (i, gp) in enumerate(gps)
		# Scale factor
		push!(post_scale_factors, σ_noise/sqrt(1. + 2. / gp.nobs))
		# logNoise
		push!(logNoise, gp.logNoise.value)
		# RKHS Norm Bound
		σ_inf = sqrt(gp.kernel.σ2*exp(-1/2*(diameter_domain)^2/gp.kernel.ℓ2))
		push!(RKHS_norm_bounds, sup_f[i] / σ_inf)
		# γ Bound
		σ_v2 = (1 + 2/(gp.nobs))
		push!(γ_bds, 0.5*gp.nobs*log(1+1/σ_v2))
	end
	Kinv = inv(gps[1].cK.mat + exp(gps[1].logNoise.value)^2*I)
    gp_info = GPRelatedInformation(γ_bds, RKHS_norm_bounds, logNoise, post_scale_factors, Kinv, sup_f, !process_noise, process_noise)
    return gp_info
end

"""
	chowdhury_rkhs_prob

Calculates the probability of the dynamics being epsilon-close given the GP parameters.
"""
function chowdhury_rkhs_prob(σ, ϵ, γ_bd, B, logNoise, post_scale_factor)
    R = post_scale_factor*exp(logNoise)
    frac = ϵ/(post_scale_factor*σ)
	if frac > B
    	dbound = exp(-0.5*(1/R*(frac - B))^2 + γ_bd + 1.)
	else
		dbound = 1.0
	end
    return minimum([dbound, 1.])
end

function chowdhury_rkhs_prob_vector(gp_rkhs_info::GPRelatedInformation, σ_bounds, ϵ; local_RKHS_bound=nothing, local_gp_metadata=nothing)
	p_rkhs = zeros(length(ϵ))
	
	for i=1:length(σ_bounds)
		# This calculates the probability of the dynamics being β(δ)*σ close.
		RKHS_bound = isnothing(local_RKHS_bound) ? gp_rkhs_info.RKHS_norm_bounds[i] : local_RKHS_bound[i]

		if !isnothing(local_gp_metadata)
			nobs = local_gp_metadata[3]
			σ_v2 = (1 + 2/(nobs))
			γ_bound =  0.5*nobs*log(1 + 1/σ_v2*local_gp_metadata[1][i]) 
		else
			γ_bound = gp_rkhs_info.γ_bounds[i]	
		end

		p_rkhs[i] = 1. - chowdhury_rkhs_prob(σ_bounds[i], ϵ[i], 
										   gp_rkhs_info.γ_bounds[i], 
										   RKHS_bound, 
										   gp_rkhs_info.logNoise[i], 
										   gp_rkhs_info.post_scale_factors[i])
	end
	return p_rkhs
end

function chowdhury_rkhs_prob_vector_single(gp_rkhs_info, σ_bounds, ϵ; local_RKHS_bound=nothing, local_gp_metadata=nothing)
	p_rkhs = 1.0
	
	for i=1:length(σ_bounds)
		# This calculates the probability of the dynamics being β(δ)*σ close.
		RKHS_bound = isnothing(local_RKHS_bound) ? gp_rkhs_info.RKHS_norm_bounds[i] : local_RKHS_bound[i]

		if !isnothing(local_gp_metadata)
			nobs = local_gp_metadata[3]
			σ_v2 = (1 + 2/(nobs))
			γ_bound =  0.5*nobs*log(1 + 1/σ_v2*local_gp_metadata[1][i]) 
		else
			γ_bound = gp_rkhs_info.γ_bounds[i]	
		end

		p_rkhs *= 1. - chowdhury_rkhs_prob(σ_bounds[i], ϵ, 
										   gp_rkhs_info.γ_bounds[i], 
										   RKHS_bound, 
										   gp_rkhs_info.logNoise[i], 
										   gp_rkhs_info.post_scale_factors[i])
	end
	return p_rkhs
end

function create_gp_info_dict(gp_info::GPRelatedInformation)
	gp_info_dict = Dict(
		"γ_bounds" => gp_info.γ_bounds,
		"RKHS_norm_bounds" => gp_info.RKHS_norm_bounds,
		"logNoise" => gp_info.logNoise,
		"post_scale_factors" => gp_info.post_scale_factors,
		"Kinv" => gp_info.Kinv,
		"f_sup" => gp_info.f_sup,
		"measurement_noise" => gp_info.measurement_noise,
		"process_noise" => gp_info.process_noise
	)
	return gp_info_dict
end
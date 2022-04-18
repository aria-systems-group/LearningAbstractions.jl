# Functions related to the RKHS stuff

struct GPRelatedInformation  # One for each set of GPs
	γ_bounds	
	RKHS_norm_bounds
	logNoise	
	post_scale_factors
	Kinv
end

"""
Create the GP Info structure componenets.
"""
function create_gp_info(gps, σ_noise, diameter_domain, sup_f)
	γ_bds = []
	RKHS_norm_bounds = []
	logNoise = []
	post_scale_factors = []
	for gp in gps
		# Scale factor
		push!(post_scale_factors, σ_noise/sqrt(1. + 2. / gp.nobs))
		# logNoise
		push!(logNoise, gp.logNoise.value)
		# RKHS Norm Bound
		σ_inf = sqrt(gp.kernel.σ2*exp(-1/2*(diameter_domain)^2/gp.kernel.ℓ2))
		push!(RKHS_norm_bounds, sup_f / σ_inf)
		# γ Bound
		B = 1 + (1 + 2/(gp.nobs))^(-1)
		push!(γ_bds, 0.5*gp.nobs*log(B))
		# K Inverse 
		push!
	end
	Kinv = inv(gps[1].cK.mat + exp(gps[1].logNoise.value)^2*I)
    gp_info = GPRelatedInformation(γ_bds, RKHS_norm_bounds, logNoise, post_scale_factors, Kinv)
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
        dbound = 1. 
    end
    return minimum([dbound, 1.])
end

function chowdhury_rkhs_prob_vector(gp_rkhs_info, σ_bounds, ϵ)
	p_rkhs = 1.
	for i=1:length(σ_bounds)
		p_rkhs *= 1. - chowdhury_rkhs_prob(σ_bounds[i], ϵ, 
										   gp_rkhs_info.γ_bounds[i], 
										   gp_rkhs_info.RKHS_norm_bounds[i], 
										   gp_rkhs_info.logNoise[i], 
										   gp_rkhs_info.post_scale_factors[i])
	end
	return p_rkhs
end
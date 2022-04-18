# GP helper functions based on GaussianProcesses.jl package using SE kernel

"""
    condition_gp_1dim

Condition a Gaussian process with the SE kernel with an N-dimensional input.
"""
function condition_gp_1dim(input, output; se_params=[0., 0.65], optimize_hyperparameters=false, lnoise=nothing, opt_fraction=1.0)
    m_prior = MeanZero()
    k_prior = SE(se_params[2], se_params[1])

    if isnothing(lnoise)
        lnoise = log(sqrt(1+2/size(input, 2))) # Generalize to handle any bound
    end

    if optimize_hyperparameters && opt_fraction > 0
        @assert opt_fraction <= 1
        num_opt = Int(opt_fraction*length(input))
        opt_idx = StatsBase.sample(1:length(output), num_opt, replace = false)
        gp_pre = GP(input[:, opt_idx], output[opt_idx], m_prior, k_prior, lnoise) 
        optimize!(gp_pre)
        gp = update_gp_data(gp_pre, input, output)
    else
        gp = GP(input, output, m_prior, k_prior, lnoise)
        if optimize_hyperparameters
            optimize!(gp)
        end 
    end

    return gp
end

"""
    condition_gps

Condition one Gaussian processe for each component of the output data.
"""
function condition_gps(input, output; se_params=[0., 0.65], optimize_hyperparameters=false, lnoise=nothing, opt_fraction=1.0)
    # gp_set = Dict()
    ndims = size(input,1)
    gps = []
    # dim_keys = ["x$i" for i=1:size(input,1)]
    for i=1:ndims 
        # Handle data dependency here
        # x_train_sub = x_train[:, findall(.>(0), data_deps[out_dim])[:]]
        x_train_sub = input
        gp = condition_gp_1dim(input, output[i,:]; se_params=se_params, optimize_hyperparameters=optimize_hyperparameters, lnoise=lnoise, opt_fraction=opt_fraction)
        
        # gp_set["x$i"] = deepcopy(gp)
        push!(gps, gp)
    end
    return gps
end


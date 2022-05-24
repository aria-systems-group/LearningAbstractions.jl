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
function condition_gps(input, output; se_params=[0., 0.65], optimize_hyperparameters=false, lnoise=nothing, opt_fraction=1.0, data_subset=0)
    # gp_set = Dict()
    ndims = size(input,1)
    gps = []
    subset_idx = data_subset > 0 ? rand(1:size(input,2), data_subset) : 1:size(input,2)
    # dim_keys = ["x$i" for i=1:size(input,1)]
    for i=1:ndims 
        # Handle data dependency here
        # x_train_sub = x_train[:, findall(.>(0), data_deps[out_dim])[:]]
        x_train_sub = input[:, subset_idx]
        y_train_sub = output[i, subset_idx]
        gp = condition_gp_1dim(x_train_sub, y_train_sub, se_params=se_params, optimize_hyperparameters=optimize_hyperparameters, lnoise=lnoise, opt_fraction=opt_fraction)
        
        # gp_set["x$i"] = deepcopy(gp)
        push!(gps, gp)
    end
    return gps
end

"""
Select k-nearest datapoints
"""
function get_local_data_knn(center, x_data, y_data; num_neighbors = 50, kdtree=nothing)
    num_neighbors = minimum([num_neighbors, size(x_data,2)])

    if isnothing(kdtree)
        kdtree = KDTree(x_data);
    end

    sub_idx, _ = knn(kdtree, center, num_neighbors, false)

    if typeof(sub_idx) == Array{Array{Int64,1},1}
        sub_idx = sub_idx[1]
    end

    sub_x_data = x_data[:, sub_idx]
    sub_y_data = y_data[sub_idx]
    return sub_x_data, sub_y_data 
end

"""
Create local GPs using k-nearest neighbors to select data.
"""
function create_local_gps(input, output, center; num_neighbors=75, kernel_params=[0., 0.65], kdtree=kdtree)
    new_gps = []
    # TODO: Simplify this with the new GP function
    for i=1:size(input, 1)
        x_nn, y_nn = get_local_data_knn(center, input, output[i,:], num_neighbors=num_neighbors, kdtree=kdtree)
        # TODO: Handle params in a better way
        push!(new_gps, condition_gp_1dim(x_nn, y_nn; se_params=kernel_params))
    end

    return new_gps
end
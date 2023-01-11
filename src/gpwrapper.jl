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
    odims = size(output,1)  # > get the dims from the output. 
    gps = []
    subset_idx = data_subset > 0 ? rand(1:size(input,2), data_subset) : 1:size(input,2)
    # dim_keys = ["x$i" for i=1:size(input,1)]
    for i=1:odims 
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
Distance Metric for SE(2)
"""
struct SE2WeightedEuclidean <: Distances.Metric
end

# TODO: Need a way to adjust these weights easily
function se2_weighted_metric(a, b; w=[1,1,pi])
    ttl = sum((a[1:2].-b[1:2]).^2 .* w[1:2])
    ttl += min(abs(a[3]- b[3]), abs(2*pi - abs(a[3] - b[3])))/w[3]
    return sqrt(ttl)
end
evaluate(dist::SE2WeightedEuclidean, a, b) = se2_weighted_metric(a, b)

"""
Distance Metric for SE(2) + R
"""
struct SE2RWeightedEuclidean <: Distances.Metric
end
# > Custom specification of radial domains
# TODO: Need a way to adjust these weights easily
function se2R_weighted_metric(a, b; w=[1,1,pi,1])
    ttl = sum((a[1:2].-b[1:2]).^2 .* w[1:2]) + (a[4] - b[4])^2*w[4]
    ttl += min(abs(a[3]- b[3]), abs(2*pi - abs(a[3] - b[3])))/w[3]
    return sqrt(ttl)
end
evaluate(dist::SE2RWeightedEuclidean, a, b) = se2R_weighted_metric(a, b)


"""
Create a tree based using the appropriate metric.
"""
function create_data_tree(input_data, domain_type) 
    if domain_type == "se(2)"
        tree = BallTree(input_data, SE2WeightedEuclidean())
    elseif domain_type == "se(2)+R"
        tree = BallTree(input_data, SE2RWeightedEuclidean())
    else    # default to pure Euclidean distance
        tree = KDTree(input_data)
    end
    return tree
end

"""
Select k-nearest datapoints
"""
function get_local_data_knn(center, x_data, y_data; num_neighbors = 50, tree=nothing)
    num_neighbors = minimum([num_neighbors, size(x_data,2)])

    if isnothing(tree)
        # TODO: domain type handling
        domain_type = ""
        tree = create_data_tree(input_data, domain_type) 
    end

    sub_idx, _ = knn(tree, center, num_neighbors, false)

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
function create_local_gps(input, output, center; num_neighbors=75, kernel_params=[0., 0.65], tree=tree)
    new_gps = []
    for i in axes(output, 1)
        x_nn, y_nn = get_local_data_knn(center, input, output[i,:], num_neighbors=num_neighbors, tree=tree)
        push!(new_gps, condition_gp_1dim(x_nn, y_nn; se_params=kernel_params))
    end

    return new_gps
end

"""
    save_gps

Save GPs with serialization.
"""
function save_gps(gps_dict, filename)
    open(filename, "w") do f
        serialize(f, gps_dict)
    end
end

"""
    load_gps

Load GPs from file.
"""
function load_gps(filename)
    f = open(filename, "r")
    gps_dict = deserialize(f)
    close(f)
    return gps_dict[:gps], gps_dict[:info]
end
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
    odims = size(output,1)  # > get the dims from the output. 
    gps = []
    subset_idx = data_subset > 0 ? sample(1:size(input,2), data_subset, replace=false) : 1:size(input,2)
    # dim_keys = ["x$i" for i=1:size(input,1)]
    for i=1:odims 
        # Handle data dependency here
        # x_train_sub = x_train[:, findall(.>(0), data_deps[out_dim])[:]]
        x_train_sub = input[:, subset_idx]
        y_train_sub = output[i, subset_idx]
        gp = condition_gp_1dim(x_train_sub, y_train_sub, se_params=se_params, optimize_hyperparameters=optimize_hyperparameters, lnoise=lnoise, opt_fraction=opt_fraction)
        
        push!(gps, gp)
    end
    return gps
end

"""
Distance Metric for General Euclidean with Angles
"""
struct GeneralMetric <: Distances.Metric
    angle_dims
    weights
end

function general_metric(a, b, dist::GeneralMetric)

    dims = length(a)
    eucl_dims = setdiff(1:dims, dist.angle_dims)
    ttl = 0
    for i in eucl_dims
        ttl += dist.weights[i]*(a[i] - b[i])^2
    end

    for j in dist.angle_dims
        ttl += min(abs(a[j]- b[j]), abs(2*pi - abs(a[j] - b[j])))*dist.weights[j]
    end

    return sqrt(ttl)
end
evaluate(dist::GeneralMetric, a, b) = general_metric(a, b, dist)

"""
Create a tree using the specified metric.
"""
function create_data_tree(input_data, general_metric::Distances.Metric) 
    tree = BallTree(input_data, general_metric)
    return tree
end

"""
Select k-nearest datapoints
"""
function get_local_data_knn(center, x_data, y_data; num_neighbors = 50, tree=nothing)
    num_neighbors = minimum([num_neighbors, size(x_data,2)])

    if isnothing(tree)
        metric = Euclidean()
        tree = create_data_tree(input_data, metric) 
    end

    sub_idx, _ = knn(tree, center, num_neighbors, false)
    sort!(sub_idx)

    if typeof(sub_idx) == Array{Array{Int64,1},1}
        sub_idx = sub_idx[1]
    end

    sub_x_data = x_data[:, sub_idx]
    sub_y_data = y_data[sub_idx]
    return sub_x_data, sub_y_data, sub_idx 
end

"""
Select k-nearest datapoints
"""
function select_knn_idxs(center, tree, num_neighbors)
    sub_idx, _ = knn(tree, center, num_neighbors, false)
    if typeof(sub_idx) == Array{Array{Int64,1},1}
        sub_idx = sub_idx[1]
    end
    return sort!(sub_idx)
end

"""
Select k-nearest datapoints using preallocated arrays for the resulting distances and indeces
"""
function select_knn_idxs!(idxs, dists, center, tree)
    NearestNeighbors.knn_point!(tree, center, false, dists, idxs, NearestNeighbors.always_false)
    sort!(idxs)
    return idxs
end


"""
Create local GPs using k-nearest neighbors to select data.
"""
function create_local_gps(input, output, center; num_neighbors=75, kernel_params=[0., 0.65], lnoise=nothing, tree=nothing)
    new_gps = []
    sub_idx = nothing 
    for i in axes(output, 1)
        x_nn, y_nn, sub_idx = get_local_data_knn(center, input, output[i,:], num_neighbors=num_neighbors, tree=tree)
        push!(new_gps, condition_gp_1dim(x_nn, y_nn; se_params=kernel_params, lnoise=lnoise))
    end

    return new_gps, sub_idx
end


"""
Create local GPs using preallocations for components via k-nearest neighbors data selection.

    Note that the k in kNN is determined implicitly from the size of the arrays in `preallocs[i].sub_idxs` and `preallocs[i].knn_dists`
"""
function create_local_gps!(preallocs, global_gps, center, tree, x, y)
    for i in eachindex(global_gps)
        # Get k nearest datapoints
        @views select_knn_idxs!(preallocs.sub_idxs, preallocs.knn_dists, center, tree)
        @views preallocs.gps[i].x[:] = x[:, preallocs.sub_idxs]
        # Update the cK matrix with local data
        update_cK_SE!(preallocs.gps[i].cK, preallocs.gps[i].x, global_gps[i].kernel.σ2, global_gps[i].kernel.ℓ2; σgp2=exp(global_gps[i].logNoise.value)^2)
        # Calculate cholesky decomp. of cK in place
        preallocs.gps[i].cKchol[:] = preallocs.gps[i].cK
        cholesky!(preallocs.gps[i].cKchol) 
        preallocs.gps[i].cKcholut.data[:] = preallocs.gps[i].cKchol
        # Calculate the inverse of cK and α vector
        preallocs.gps[i].K_inv.data[:] = preallocs.gps[i].cK
        inv!(preallocs.gps[i].K_inv)
        mul!(preallocs.gps[i].alpha, preallocs.gps[i].K_inv, y[i, preallocs.sub_idxs])
    end
    return 
end

"""
    inv!(A::Union{Symmetric{<:BlasReal},Hermitian{<:BlasFloat}})

Inplace, zero-allocation inverse for a hermitian matrix.
From: https://discourse.julialang.org/t/non-allocating-matrix-inversion/62264
"""
function inv!(A::Union{Symmetric{<:BlasReal},Hermitian{<:BlasFloat}})
    _, info = LAPACK.potrf!(A.uplo, A.data)
    (info == 0) || throw(PosDefException(info))
    LAPACK.potri!(A.uplo, A.data)
    return A
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

"""
SE kernel function.
"""
function se_k(x,y; σ2=1., ℓ2=1.)
    nr = 0
    for i in eachindex(x)
       nr += (x[i] - y[i])^2 
    end
    return σ2*exp(-nr/(2. * ℓ2))
end

"""
Calculate the covariance matrix using the SE kernel.
"""
function update_cK_SE!(cK, x, σ2, ℓ2; σgp2=1e-4)
    for i in axes(cK, 2)
        for j = i:size(cK, 1)
            @views cK[i,j] = se_k(x[:,i], x[:,j], σ2=σ2, ℓ2=ℓ2)
            @views cK[j,i] = cK[i,j]
        end
    end

    for i in axes(cK, 2)
        cK[i,i] += σgp2
    end
    cK[:,:] = Symmetric(cK)
    return cK
end
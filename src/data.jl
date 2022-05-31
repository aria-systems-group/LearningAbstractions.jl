# Contains functions for generating synthetic training data
module DataGen

using TOML

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)
using Distributions
using Random

using Printf
using BSON

# using StaticArrays

function generate_system_data(config_file::String)
    f = open(config_file)
	config = TOML.parse(f)
	close(f)

    dataset_dict = Dict()

    # ! Use at your own risk!
    f_ex = Meta.parse(config["system"])
    dyn_fcn = @RuntimeGeneratedFunction(f_ex)

    noise_config = config["noise"]
    if noise_config["measurement_distribution"] == "Gaussian"
        measurement_noise_dist = Normal(noise_config["measurement_mean"], noise_config["measurement_std"])
        dataset_dict[:measurement_noise] = noise_config 
    else
        measurement_noise_dist = nothing
    end

    data_config = config["data"]
    sample_range = data_config["sample_range"]
    dataset_size = data_config["dataset_size"]
    input, output = DataGen.sample_function(dyn_fcn, sample_range, dataset_size, measurement_noise_dist=measurement_noise_dist)
    dataset_dict[:size] = dataset_size
    dataset_dict[:input] = input
    dataset_dict[:output] = output

    if data_config["save_data"] 
        system_tag = config["system_tag"]
        if !isnothing(measurement_noise_dist) 
            m_noise_tag = @sprintf "%s_%1.1f_%1.2f" noise_config["measurement_distribution"] noise_config["measurement_mean"] noise_config["measurement_std"] 
        else 
            m_noise_tag = nothing
        end

        data_filename = @sprintf "data-%s-size_%d" system_tag dataset_size
        data_filename = isnothing(m_noise_tag) ? data_filename : @sprintf "%s-%s" data_filename m_noise_tag
        data_filename = "$data_filename.bson"
        dataset_dict[:filename] = data_filename
        BSON.@save(data_filename, dataset_dict)
    end

    return dataset_dict
end
"""
    sample_function

Samples the given function from uniform random input datapoints in the hyperrectangle.
"""
function sample_function(fcn, range::Vector, num_samples::Int; 
                         random_seed::Int64=11, process_noise_dist=nothing, measurement_noise_dist=nothing)

    n_dims_in = length(range[1])
    n_dims_out = length(fcn(range[1]))
    mt = MersenneTwister(random_seed)

    input = vcat([rand(mt, Uniform(range[1][i], range[2][i]), 1, num_samples) for i=1:n_dims_in]...)
    output = mapslices(fcn, input, dims=1) 

    if !isnothing(process_noise_dist)
        output += rand(mt, process_noise_dist, (n_dims_out, num_samples)) 
    end
    if !isnothing(measurement_noise_dist)
        output += rand(mt, measurement_noise_dist, (n_dims_out, num_samples)) 
    end

    @assert size(input, 2) == size(output, 2)
    return input, output
end

function remove_known_function(fcn, input, output)
    return output - mapslices(fcn, input, dims=1)
end

end
# Tools that facilitate IMDP verification.

function is_point_in_rectangle(pt, rectangle)
	dim = length(pt)
	# Format of rectange: [lowers, uppers]
	res = true
	for i=1:dim
		res = rectangle[i] <= pt[i] <= rectangle[i+dim]
		if !res 
			return false
		end
	end
	return true
end

function create_imdp_labels(labels_fn, imdp, all_state_means)
    for i = eachindex(all_state_means) 
        imdp.labels[i] = labels_fn(all_state_means[i])
    end
    imdp.labels[length(imdp.states)] = labels_fn(nothing, unsafe=true)
end

"""
    general_label_fcn

Function prototype to label IMDP states.
"""
function general_label_fcn(point, default_label::String, unsafe_label::String, labels_dict::Dict; unsafe=false, unsafe_default=false)
    if unsafe 
        # Hacky workaround
        if unsafe_default
            return default_label
        end
        return unsafe_label 
    end
    state_label = default_label
    for label in keys(labels_dict) 
        for region in labels_dict[label]
            if is_point_in_rectangle(point, region) 
                state_label = label
                break
            end
        end
    end
    return state_label
end

"""
    load_PCTL_specification

Load a PCTL specification from a TOML file.
"""
function load_PCTL_specification(spec_filename::String)
    f = open(spec_filename)
    spec_data = TOML.parse(f)
    close(f)

    ϕ1 = spec_data["phi1"] == false ? nothing : spec_data["phi1"] 
    ϕ2 = spec_data["phi2"]
    default_label = spec_data["default"]
    unsafe_label = spec_data["unsafe"]

    labels_dict = Dict(ϕ1 => [], ϕ2 => [], unsafe_label => [])
    dims = spec_data["dims"]
    for geometry in spec_data["labels"]["phi1"]
        push!(labels_dict[ϕ1], geometry)
    end

    for geometry in spec_data["labels"]["phi2"]
        push!(labels_dict[ϕ2], geometry)
    end

    for geometry in spec_data["labels"]["unsafe"]
        push!(labels_dict[unsafe_label], geometry)
    end

    unsafe_default_flag = spec_data["default_outside_compact"]

    lbl_fcn = (point; unsafe=false) -> general_label_fcn(point, default_label, unsafe_label, labels_dict, unsafe=unsafe, unsafe_default=unsafe_default_flag)
    return lbl_fcn, labels_dict, ϕ1, ϕ2, spec_data["steps"], spec_data["name"]
end

function load_environment_labels(environemnt_filename::String)
    f = open(environemnt_filename)
    environmenet_data = TOML.parse(f)
    close(f)

    labels = environmenet_data["labels"]
    labels_dict = Dict()
    for label in keys(labels)
        labels_dict[label] = []
        for geometry in labels[label]
            push!(labels_dict[label], geometry)
        end
    end

    default_label = ""
    unsafe_label = "unsafe"
    label_fcn = (x; unsafe=false) -> general_label_fcn(x, default_label, unsafe_label, labels_dict, unsafe=unsafe, unsafe_default=false)
    return label_fcn, labels_dict
end
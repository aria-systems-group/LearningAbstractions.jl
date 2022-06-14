# Tools that facilitate IMDP verification.

function create_imdp_labels(labels_fn, imdp, all_state_means)
    for i = 1:length(all_state_means) 
        point = Meshes.Point(all_state_means[i])
        imdp.labels[i] = labels_fn(point)
    end
    imdp.labels[length(imdp.states)] = labels_fn(nothing, unsafe=true)
end

"""
    general_label_fcn

Function prototype to label IMDP states.
"""
function general_label_fcn(point, default_label::String, unsafe_label::String, labels_dict::Dict; unsafe=false)
    if unsafe # !wtf is this
        return unsafe_label 
    end
    state_label = default_label
    for label in keys(labels_dict) 
        for region in labels_dict[label]
            if point ∈ region
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
        push!(labels_dict[ϕ1], Box(Point(geometry[1:dims]), Point(geometry[dims+1:end])))
    end

    for geometry in spec_data["labels"]["phi2"]
        push!(labels_dict[ϕ2], Box(Point(geometry[1:dims]), Point(geometry[dims+1:end])))
    end

    for geometry in spec_data["labels"]["unsafe"]
        push!(labels_dict[unsafe_label], Box(Point(geometry[1:dims]), Point(geometry[dims+1:end])))
    end

    lbl_fcn = (point; unsafe=false) -> general_label_fcn(point, default_label, unsafe_label, labels_dict, unsafe=unsafe)
    return lbl_fcn, labels_dict, ϕ1, ϕ2, spec_data["steps"], spec_data["name"]
end


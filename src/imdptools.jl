

function create_imdp_labels(labels_fn, imdp, all_state_means)
    for i = 1:length(all_state_means) 
        point = Meshes.Point(all_state_means[i])
        imdp.labels[i] = labels_fn(point)
    end
end

# ! types on this function are fucked
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

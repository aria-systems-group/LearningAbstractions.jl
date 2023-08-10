# Tools that facilitate IMDP verification.

function post(state_idx, P̂)
    return findall(x -> x>0., P̂[state_idx, :])
end

function pre(state_idx, P̂)
    return findall(x -> x>0., P̂[:, state_idx])
end

function true_transition_propabilities(pmin::AbstractVector, pmax::AbstractVector, indeces::AbstractVector)

    @assert length(pmin) == length(pmax) == length(indeces)

    p = zeros(size(pmin))
    used = sum(pmin)
    remain = 1 - used

    for i in indeces
        if pmax[i] <= (remain + pmin[i])
            p[i] = pmax[i]
        else
            p[i] = pmin[i] + remain
        end
        remain = max(0, remain - (pmax[i] - pmin[i]))
    end

    return p
end
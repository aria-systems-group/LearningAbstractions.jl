# Tools that facilitate IMDP verification.

function post(state_idx, P̂)
    return findall(x -> x>0., P̂[state_idx, :])
end

function pre(state_idx, P̂)
    return findall(x -> x>0., P̂[:, state_idx])
end
function pre_emphasis_filter(x, coeff=0.95f0)
    return vcat(x[1,:,:], x[2:end, :, :] - coeff * x[1:end-1, :, :])
end

function error_to_signal(ŷ, y)
    #Error to signal ratio with pre-emphasis filter:
    #https://www.mdpi.com/2076-3417/10/3/766/htm

    y = pre_emphasis_filter(y) 
    ŷ = pre_emphasis_filter(ŷ)

    return sum((y - ŷ).^2)  ./ (sum(y.^2) .+ 1e-10)
end



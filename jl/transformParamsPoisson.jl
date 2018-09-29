"""
For transforming the mean of a parent into the multiscale parameters for the children.

Arguments:  coarse_mean, the parent's mean
            fine_means, the childens' means

Returns:    omega_j,k
"""
function transformParamsPoisson(coarse_mean::Float64, fine_means::Array{Float64,1})
  omegas = fine_means/coarse_mean;
  return omegas
end

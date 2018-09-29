"""
For transforming the mean and variance of a parent and its children into the multiscale parameters for the children.

Arguments:  coarse_mean, the parent mean
            coarse_var, the parent variance
            fine_means, the childrens' means
            fine_vars, the childrens' variances

Returns:    nu_j,k, omega_j,k, Omega_j,k
"""
function transformParamsGaussian(coarse_mean::Float64, coarse_var::Float64, fine_means::Array{Float64,1}, fine_vars::Array{Float64,1})
  nus = fine_vars/coarse_var;
  omegas = fine_means - nus*coarse_mean;
  big_omega = diagm(fine_vars) - (fine_vars .* fine_vars')/coarse_var;
  return nus, omegas, big_omega
end

module Multiscale

using Distributions

export getFinestDistributionGaussian,
  transformParamsGaussian,
  transformParamsPoisson

include("getFinestDistributionGaussian.jl")
include("transformParamsGaussian.jl")
include("transformParamsPoisson.jl")

end

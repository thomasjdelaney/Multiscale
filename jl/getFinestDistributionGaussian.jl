"""
For getting the model probability distribution at the finest level, given the parameters for each level and a measurement from each of the coarser levels.

Arguments:  level_params, An array containing all the parameters of the lowel levels
            level_measures, an array containing measures from the coarses levels
            num_levels, the number of levels

Returns:    probability distribution, a product of Gaussians.
"""
function getFinestDistributionGaussian(level_params::Array, level_measures::Array, num_levels::Int)
  checkLevelParamsDimensions(level_params::Array, level_measures::Array, num_levels::Int);
  top_distn = MultivariateNormal(level_params[1][1], level_params[1][2]);
  level_distns = Array{Any,1}(num_levels-1);
  for level in 1:(num_levels-1)
    coarse_measure = level_measures[level];
    num_distns = length(level_measures[level]);
	level_distns[level] = Array{Distributions.Distribution,1}(num_distns);
    if num_distns == 1
      level_nus, level_omegas, level_big_omega = level_params[level+1];
      level_distns[level] = MultivariateNormal(level_nus*coarse_measure + level_omegas, level_big_omega);
    else
      for d in 1:num_distns
        level_nus, level_omegas, level_big_omega = level_params[level+1][d];
        level_distns[level][d] = MultivariateNormal(level_nus*coarse_measure[d] + level_omegas, level_big_omega)
      end
    end
  end
end

function checkLevelParamsDimensions(level_params::Array, level_measures::Array, num_levels::Int)
  if size(level_params,1) != (num_levels)
    error("size(level_params,1) != (num_levels)");
  end
  if size(level_measures,1) != (num_levels-1)
    error("size(level_measures,1) != (num_levels-1)");
  end
  if size(level_params[1],1) != 2
	error("size(level_params[1],1) != 2");
  end
  for i in 2:(num_levels-1)
    if size(level_params[i][1],1) != size(level_params[i+1],1)
      error("size(level_params[i][1],1) != size(level_params[i+1],1)");
    end
  end
  if size(level_measures[1],1) != 1
	error("size(level_measures[1],1) != 1");
  end
  for i in 2:(num_levels-1)
	if size(level_measures[i],1) != size(level_params[i][1],1)
	  error("size(level_measures[$i],1) != size(level_params[$i][1],1)");
  	end
  end
end

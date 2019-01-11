Project for fitting a multiscale model as described in Kolaczyk et al 2000.

Python 2.7.15

###### Required Packages:

1. pandas 0.23.4
2. random
3. numpy 1.15.2
4. anytree 2.4.3
5. scipy 1.1.0
6. sklearn 0.20.0

#### Recovering parameters via hierarchical model

In order to recover the means of the finest level measurements for either the Gaussian or Poisson hierarchical model, run:
* python -i py/recover_synthetic_gaussian_parameters.py
* python -i py/recover_synthetic_poisson_parameters.py
The true parameters can be found in the 'region_param_frame' pandas dataframe, the estimated parameters can be found in the  'region_param_estimated' pandas dataframe.

TODO:
* Save and load tree using YAML.
* calculate marginal likelihood of more than one hierarchy of partitions

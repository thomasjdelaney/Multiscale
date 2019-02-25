Project for fitting a multiscale model as described in Kolaczyk et al 2000.

Python 3.7.2

###### Required Packages:

1. pandas 0.24.1
2. random
3. numpy 1.16.1
4. anytree 2.6.0
5. scipy 1.2.1
6. sklearn 0.20.2

#### Recovering parameters via hierarchical model

In order to recover the means of the finest level measurements for either the Gaussian or Poisson hierarchical model, run:
```python
python -i py/recover_synthetic_gaussian_parameters.py
```
or
```python
python -i py/recover_synthetic_poisson_parameters.py
```
The true parameters can be found in the 'region_param_frame' pandas dataframe, the estimated parameters can be found in the  'region_param_estimated' pandas dataframe.

#### Creating synthetic data

To create a csv file containing heirarchical synthetic data, run:
```python
python py/synthesise_hierarchical_gaussian_data.py
```
or
```python
python py/synthesise_hierarchical_poisson_data.py
```

The files will be saved in the 'csv' directory.

#### Creating synthetic data with given correlations

It is possible to create synthetic hierarchical data with correlations chosen by the user. Just run:
```python
python py/synthesise_hierarchical_correlated_gaussian_data.py --covariance_type manual --correlation_values 0.5 0.3 0.1 --save_prefix man_
```
The csv files will be saved in the 'csv' directory and the 'save_prefix' will be fixed to the beginning of the files' names. The three 'correlation_values' correspond to intraprovincial correlation, intracountry (but extraprovincial) correlation, and extracountry correlation.

#### Examples with plots

For data from independent regions, run:
```python
python py/synthesise_hierarchical_correlated_gaussian_data.py --covariance_type independent --save_prefix indy_ --num_samples 1000
python -i py/recover_synthetic_gaussian_parameters.py --plot_mean_accuracy --plot_correlation --plot_variance_accuracy --csv_file_prefix indy_ --num_samples 1000
```
For data with manually controlled correlations, run:
```python
python py/synthesise_hierarchical_correlated_gaussian_data.py --covariance_type manual --correlation_values 0.5 0.3 0.1 --save_prefix man_ --num_samples 1000
python -i py/recover_synthetic_gaussian_parameters.py --plot_mean_accuracy --plot_correlation --plot_variance_accuracy --csv_file_prefix man_ --num_samples 1000
```
#### Using different trees

Examples of trees can be found in ```py/true_tree.py``` and ```py/false_tree.py```. At the moment, every valid tree must have 2 countries, 6 provinces, and 24 regions. The tree used to create synthetic data is ```true_tree.py```, the 'false' tree can be used to measure differences created by using an incorrect tree. To make use of a given tree, use the ```--tree_file``` command line argument for the ```recover_synthetic_gaussian_parameters.py``` script.



###### TODO:
* Save and load tree using YAML.
* calculate marginal likelihood of more than one hierarchy of partitions
* A lot of variance is pushed into the 'leftover' regions. Why is that?

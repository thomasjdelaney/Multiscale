"""
Recover the parameters used to create the hierarchical data by calculating the multiscale parameters,
and inference.
Useful line for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from sklearn.datasets import make_spd_matrix # for generating random covariance matrices

pd.set_option('max_rows', 30)
# defining directories
proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv', 'poisson')

# loading hierarchical tree structure and province colours
execfile(os.path.join(py_dir, 'tree_and_colours.py'))

# loading measurements
country_measurement_frame = pd.read_csv(os.path.join(csv_dir, 'country_measurement_frame.csv'), index_col='row_num')
province_measurement_frame = pd.read_csv(os.path.join(csv_dir, 'province_measurement_frame.csv'), index_col='row_num')
region_measurement_frame = pd.read_csv(os.path.join(csv_dir, 'region_measurement_frame.csv'), index_col='row_num')
# loading ground truth
country_param_frame = pd.read_csv(os.path.join(csv_dir, 'country_param_frame.csv'), index_col='parameter')
province_param_frame = pd.read_csv(os.path.join(csv_dir, 'province_param_frame.csv'), index_col='parameter')
region_param_frame = pd.read_csv(os.path.join(csv_dir, 'region_param_frame.csv'), index_col='parameter')

def getEstimatedParamsForPartition(partition_measurements):
    partition_param_estimated = pd.DataFrame()
    partition_mean_estimated = partition_measurements.mean()
    partition_mean_estimated.name = 'mean'
    partition_param_estimated = partition_param_estimated.append(partition_mean_estimated)
    return partition_param_estimated

def getNodeExpectedOmega(node, node_param_estimated, child_param_estimated):
    num_children = len(node.children)
    child_names = [c.name for c in node.children]
    child_means = child_param_estimated.loc['mean'][child_names]
    node_mean = node_param_estimated.loc['mean'][node.name]
    node_gamma = np.random.rand()
    omega_hat = (node_gamma*np.ones(num_children) + child_means)/(num_children*node_gamma + node_mean)
    return omega_hat

def getPartitionOmegas(partition_nodes, partition_param_estimated, child_param_estimated):
    child_node_names = child_param_estimated.columns
    child_omega = pd.concat([getNodeExpectedOmega(node, partition_param_estimated, child_param_estimated) for node in partition_nodes])
    child_param_estimated = child_param_estimated.append(pd.Series(child_omega, index=child_node_names, name='omega'))
    return child_param_estimated

def getHierarchicalMeanEstimatebyNode(node, node_param_estimated, child_param_estimated):
    child_names = [c.name for c in node.children]
    node_mean = node_param_estimated.loc['mean'][node.name]
    child_omega = child_param_estimated.loc['omega'][child_names]
    return child_omega * node_mean

def getHierarchicalMeanEstimate(partition_nodes, node_param_estimated, child_param_estimated):
    mean_estimates = pd.concat(getHierarchicalMeanEstimatebyNode(node, node_param_estimated, child_param_estimated) for node in partition_nodes)
    mean_estimates.name = 'hier_mean'
    return child_param_estimated.append(mean_estimates)

country_param_estimated = getEstimatedParamsForPartition(country_measurement_frame)
province_param_estimated = getEstimatedParamsForPartition(province_measurement_frame)
region_param_estimated = getEstimatedParamsForPartition(region_measurement_frame)

# calculating hierarchical covariances
province_param_estimated = getPartitionOmegas([country_0, country_1], country_param_estimated, province_param_estimated)
region_param_estimated = getPartitionOmegas(country_0.children + country_1.children, province_param_estimated, region_param_estimated)

# estimating hierarchical means
province_param_estimated = getHierarchicalMeanEstimate([country_0, country_1], country_param_estimated, province_param_estimated)
region_param_estimated = getHierarchicalMeanEstimate(country_0.children + country_1.children, province_param_estimated, region_param_estimated)

# reordering columns
country_param_estimated = country_param_estimated[country_param_frame.columns]
province_param_estimated = province_param_estimated[province_param_frame.columns]
region_param_estimated = region_param_estimated[region_param_frame.columns]

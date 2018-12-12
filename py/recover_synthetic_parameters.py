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
import anytree as at
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal, norm
from sklearn.datasets import make_spd_matrix # for generating random covariance matrices

pd.set_option('max_rows', 30)
# defining directories
proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
csv_dir = os.path.join(proj_dir, 'csv')
# loading measurements
country_measurement_frame = pd.read_csv(os.path.join(csv_dir, 'country_measurement_frame.csv'), index_col='row_num')
province_measurement_frame = pd.read_csv(os.path.join(csv_dir, 'province_measurement_frame.csv'), index_col='row_num')
region_measurement_frame = pd.read_csv(os.path.join(csv_dir, 'region_measurement_frame.csv'), index_col='row_num')
# loading ground truth
country_param_frame = pd.read_csv(os.path.join(csv_dir, 'country_param_frame.csv'), index_col='parameter')
province_param_frame = pd.read_csv(os.path.join(csv_dir, 'province_param_frame.csv'), index_col='parameter')
region_param_frame = pd.read_csv(os.path.join(csv_dir, 'region_param_frame.csv'), index_col='parameter')
# building trees TODO: move this to a separate script and save down using YAML
country_0 = at.Node('country_0')
country_1 = at.Node('country_1')
province_0 = at.Node('province_0', parent=country_0)
province_1 = at.Node('province_1', parent=country_0)
province_2 = at.Node('province_2', parent=country_0)
province_3 = at.Node('province_3', parent=country_1)
province_4 = at.Node('province_4', parent=country_1)
province_5 = at.Node('province_5', parent=country_1)
region_0 = at.Node('region_0', parent=province_0)
region_1 = at.Node('region_1', parent=province_0)
region_2 = at.Node('region_2', parent=province_0)
region_3 = at.Node('region_3', parent=province_1)
region_4 = at.Node('region_4', parent=province_1)
region_5 = at.Node('region_5', parent=province_1)
region_6 = at.Node('region_6', parent=province_1)
region_7 = at.Node('region_7', parent=province_2)
region_8 = at.Node('region_8', parent=province_2)
region_9 = at.Node('region_9', parent=province_2)
region_10 = at.Node('region_10', parent=province_2)
region_11 = at.Node('region_11', parent=province_2)
region_12 = at.Node('region_12', parent=province_3)
region_13 = at.Node('region_13', parent=province_3)
region_14 = at.Node('region_14', parent=province_3)
region_15 = at.Node('region_15', parent=province_3)
region_16 = at.Node('region_16', parent=province_3)
region_17 = at.Node('region_17', parent=province_4)
region_18 = at.Node('region_18', parent=province_4)
region_19 = at.Node('region_19', parent=province_4)
region_20 = at.Node('region_20', parent=province_4)
region_21 = at.Node('region_21', parent=province_5)
region_22 = at.Node('region_22', parent=province_5)
region_23 = at.Node('region_23', parent=province_5)

province_to_color = {}
colours = cm.gist_rainbow(np.linspace(0, 1, 6))
province_to_color[province_0] = colours[0]
province_to_color[province_1] = colours[1]
province_to_color[province_2] = colours[2]
province_to_color[province_3] = colours[3]
province_to_color[province_4] = colours[4]
province_to_color[province_5] = colours[5]

def getEstimatedParamsForPartition(partition_measurements):
    partition_param_estimated = pd.DataFrame()
    partition_mean_estimated = partition_measurements.mean()
    partition_mean_estimated.name = 'mean'
    partition_var_estimated = partition_measurements.var()
    partition_var_estimated.name = 'var'
    partition_param_estimated = partition_param_estimated.append(partition_mean_estimated)
    partition_param_estimated = partition_param_estimated.append(partition_var_estimated)
    return partition_param_estimated

def getNodeExpectedOmega(node, node_measure_frame, node_param_estimated, child_measure_frame, child_param_estimated):
    n = child_measure_frame.shape[0] # num samples
    num_children = len(node.children)
    child_names = [c.name for c in node.children]
    node_var = node_param_estimated.loc['var'][node.name]
    child_vars = child_param_estimated.loc['var'][child_names]
    node_nu = child_vars/node_var
    node_phi = make_spd_matrix(num_children, random_state=1)/100 # prior variance matrix
    node_Omega = np.diag(child_vars) - np.outer(child_vars, child_vars)/node_var
    post_covariance = np.linalg.inv(np.linalg.inv(node_phi) + n*np.linalg.inv(node_Omega))
    measurement_factor = child_measure_frame[child_names].mean() - node_measure_frame[node.name].mean()*node_nu
    omega_hat = n * np.matmul(post_covariance, np.matmul(np.linalg.inv(node_Omega), measurement_factor))
    return omega_hat, node_nu.values, np.diag(post_covariance)

def getPartitionOmegas(partition_nodes, partition_measure_frame, partition_param_estimated, child_measure_frame, child_param_estimated):
    child_node_names = child_param_estimated.columns
    child_omega_nu = [getNodeExpectedOmega(node, partition_measure_frame, partition_param_estimated, child_measure_frame, child_param_estimated) for node in partition_nodes]
    child_omega_estimated, child_nu, child_post_variance = np.concatenate(child_omega_nu, axis=1)
    child_param_estimated = child_param_estimated.append(pd.Series(child_omega_estimated, index=child_node_names, name='omega'))
    child_param_estimated = child_param_estimated.append(pd.Series(child_nu, index=child_node_names, name='nu'))
    return child_param_estimated.append(pd.Series(child_post_variance, index=child_node_names, name='posterior_variance'))

def getHierarchicalMeanEstimatebyNode(node, node_param_estimated, child_param_estimated):
    child_names = [c.name for c in node.children]
    node_mean = node_param_estimated.loc['mean'][node.name]
    child_nu = child_param_estimated.loc['nu'][child_names]
    child_omega = child_param_estimated.loc['omega'][child_names]
    return child_omega + node_mean*child_nu

def getHierarchicalMeanEstimate(partition_nodes, node_param_estimated, child_param_estimated):
    mean_estimates = pd.concat(getHierarchicalMeanEstimatebyNode(node, node_param_estimated, child_param_estimated) for node in partition_nodes)
    mean_estimates.name = 'hier_mean'
    return child_param_estimated.append(mean_estimates)

def plotRegionalDistnWithEstParam(region_nodes, region_param_frame, region_param_estimated):
    num_regions = len(region_nodes)
    real_fig = plt.figure(0)
    post_fig = plt.figure(1)

    def plotGaussianWithLine(gaussian, colour, line_x, plot_title):
        x_axis = np.linspace(gaussian.mean() - 3*gaussian.std(), gaussian.mean() + 3*gaussian.std(), 1000)
        plt.plot(x_axis, gaussian.pdf(x_axis), color=colour)
        plt.vlines(line_x, ymin=0, ymax=plt.ylim()[1], alpha=0.3, linestyles='dashed')
        plt.title(plot_title, fontsize='small')
        plt.xticks(fontsize='small'); plt.yticks([]);

    for i in range(0, num_regions):
        region_node = region_nodes[i]
        region_name = region_node.name
        region_mean, region_std = region_param_frame[region_name][['mean', 'std']]
        region_hier_mean, region_post_var = region_param_estimated[region_name][['hier_mean', 'posterior_variance']]
        region_post_std = np.sqrt(region_post_var)
        region_distn = norm(region_mean, region_std)
        post_distn = norm(region_hier_mean, region_post_std)
        plt.figure(0); plt.subplot(6, 4, i+1);
        plotGaussianWithLine(region_distn, province_to_color[region_node.parent], region_hier_mean, region_name.replace('_', ' '))
        plt.figure(1); plt.subplot(6, 4, i+1);
        plotGaussianWithLine(post_distn, province_to_color[region_node.parent], region_mean, region_name.replace('_', ' '))
    plt.figure(0); plt.suptitle('True distributions with hierarchically estimated means'); plt.tight_layout();
    plt.figure(1); plt.suptitle('Posterior distributions with true means'); plt.tight_layout();

country_param_estimated = getEstimatedParamsForPartition(country_measurement_frame)
province_param_estimated = getEstimatedParamsForPartition(province_measurement_frame)
region_param_estimated = getEstimatedParamsForPartition(region_measurement_frame)

# calculating hierarchical covariances
province_param_estimated = getPartitionOmegas([country_0, country_1], country_measurement_frame, country_param_estimated, province_measurement_frame, province_param_estimated)
region_param_estimated = getPartitionOmegas(country_0.children + country_1.children, province_measurement_frame, province_param_estimated, region_measurement_frame, region_param_estimated)

# estimating hierarchical means
province_param_estimated = getHierarchicalMeanEstimate([country_0, country_1], country_param_estimated, province_param_estimated)
region_param_estimated = getHierarchicalMeanEstimate(country_0.children + country_1.children, province_param_estimated, region_param_estimated)

# reordering columns
country_param_estimated = country_param_estimated[country_param_frame.columns]
province_param_estimated = province_param_estimated[province_param_frame.columns]
region_param_estimated = region_param_estimated[region_param_frame.columns]

# plotting actual distributions vs estimated means
region_nodes = province_0.children + province_1.children + province_2.children + province_3.children + province_4.children + province_5.children

"""
Recover the parameters used to create the hierarchical data by calculating the multiscale parameters,
and inference.
Useful line for editing:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import sys, argparse, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import multivariate_normal, norm
from sklearn.datasets import make_spd_matrix # for generating random covariance matrices
from decimal import Decimal # for printing numbers in scientific notation
from numpy.random import multivariate_normal

# command line arguments
parser = argparse.ArgumentParser(description='Recover the parameters used to create the hierarchical data by calculating the multiscale parameters, and inference.')
parser.add_argument('-i', '--use_random_matrix', help='Use scaled random matrix as the prior covariance matrix.', action='store_true', default=False)
parser.add_argument('-s', '--identity_scaling_value', help='The value to scale the identity matrix.', type=float, default=1.0)
parser.add_argument('-p', '--plot_mean_accuracy', help='Flag to make the mean accuracy plot.', action='store_true', default=False)
parser.add_argument('-a', '--save_mean_accuracy', help='Flag to save the scaling value and mean squared accuracy.', action='store_true', default=False)
parser.add_argument('-f', '--csv_file_prefix', help='Prefix to attach to csv file names.', type=str, default='')
parser.add_argument('-d', '--debug', help='Flag to enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

pd.set_option('max_rows', 30)
# defining directories
proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv', 'gaussian')
image_dir = os.path.join(proj_dir, 'images', 'gaussian')
results_csv_dir = os.path.join(proj_dir, 'csv', 'results')

# loading hierarchical tree structure and province colours
execfile(os.path.join(py_dir, 'tree_and_colours.py'))

# loading shared plotting functions
sys.path.append(py_dir)
import plottingMultiscale as pm

def loadMeasurementsAndTruth(csv_dir, prefix):
    # loading measurements
    country_measurement_frame = pd.read_csv(os.path.join(csv_dir, prefix + 'country_measurement_frame.csv'), index_col='row_num')
    province_measurement_frame = pd.read_csv(os.path.join(csv_dir, prefix + 'province_measurement_frame.csv'), index_col='row_num')
    region_measurement_frame = pd.read_csv(os.path.join(csv_dir, prefix + 'region_measurement_frame.csv'), index_col='row_num')
    # loading ground truth
    country_param_frame = pd.read_csv(os.path.join(csv_dir, prefix + 'country_param_frame.csv'), index_col='parameter')
    province_param_frame = pd.read_csv(os.path.join(csv_dir, prefix + 'province_param_frame.csv'), index_col='parameter')
    region_param_frame = pd.read_csv(os.path.join(csv_dir, prefix + 'region_param_frame.csv'), index_col='parameter')
    return country_measurement_frame, province_measurement_frame, region_measurement_frame, country_param_frame, province_param_frame, region_param_frame

def getEstimatedParamsForPartition(partition_measurements):
    partition_param_estimated = pd.DataFrame()
    partition_mean_estimated = partition_measurements.mean()
    partition_mean_estimated.name = 'mean'
    partition_var_estimated = partition_measurements.var()
    partition_var_estimated.name = 'var'
    partition_std_estimated = partition_measurements.std()
    partition_std_estimated.name = 'std'
    partition_param_estimated = partition_param_estimated.append(partition_mean_estimated)
    partition_param_estimated = partition_param_estimated.append(partition_var_estimated)
    partition_param_estimated = partition_param_estimated.append(partition_std_estimated)
    return partition_param_estimated

def getMeanComparisonPlotSuptitleAndFilename(identity_scale, use_random_matrix, mean_squared_difference=0.0):
    if use_random_matrix:
        suptitle = 'Random prior covariance'
        filename = 'mean_comparisons_random_covariance.png'
    else:
        suptitle = 'Prior covariance = ' + str(identity_scale) + '*I'
        filename = 'mean_comparisons_'+ str(identity_scale) + '_covariance.png'
    if mean_squared_difference > 0.0:
        suptitle = suptitle + ', Mean squared difference = ' + '%.2E' % Decimal(mean_squared_difference)
    return suptitle, filename

def getPriorCovarianceMatrix(use_random_matrix, num_children, identity_scale):
    if use_random_matrix:
        prior_covariance = make_spd_matrix(num_children, random_state=1)/100
    else:
        prior_covariance = identity_scale * np.identity(num_children)
    return prior_covariance

def getNodeOmega(node, node_param_estimated, child_param_estimated):
    node_var = node_param_estimated.loc['var'][node.name]
    child_names = [c.name for c in node.children]
    child_vars = child_param_estimated.loc['var'][child_names]
    node_nu = child_vars/node_var
    node_Omega = np.diag(child_vars) - np.outer(child_vars, child_vars)/node_var
    return child_names, node_nu, node_Omega

def getNodeExpectedOmega(node, node_measure_frame, node_param_estimated, child_measure_frame, child_param_estimated):
    n = child_measure_frame.shape[0] # num samples
    num_children = len(node.children)
    node_phi = getPriorCovarianceMatrix(args.use_random_matrix, num_children, args.identity_scaling_value)
    child_names, node_nu, node_Omega = getNodeOmega(node, node_param_estimated, child_param_estimated)
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

def sampleFromModel(province_nodes, province_param_estimated, region_param_estimated, num_samples=1000):
    model_measurement_frame = pd.DataFrame(np.zeros([num_samples, region_param_estimated.shape[1]]), columns=region_param_estimated.columns)
    for province_node in province_nodes:
        region_names, province_nu, Omega = getNodeOmega(province_node, province_param_estimated, region_param_estimated)
        child_samples = multivariate_normal(region_param_estimated[region_names].loc['hier_mean'], Omega, num_samples)
        model_measurement_frame[region_names] = child_samples
    return model_measurement_frame

def plotRegionalDistnWithEstParam(region_nodes, param_frame, param_estimated, colour_dict, num_rows=6, num_columns=4):
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
        region_mean, region_std = param_frame[region_name][['mean', 'std']]
        region_hier_mean, region_post_var = param_estimated[region_name][['hier_mean', 'posterior_variance']]
        region_post_std = np.sqrt(region_post_var)
        region_distn = norm(region_mean, region_std)
        post_distn = norm(region_hier_mean, region_post_std)
        plt.figure(0); plt.subplot(num_rows, num_columns, i+1);
        plotGaussianWithLine(region_distn, colour_dict[region_node.parent], region_hier_mean, region_name.replace('_', ' '))
        plt.figure(1); plt.subplot(num_rows, num_columns, i+1);
        plotGaussianWithLine(post_distn, colour_dict[region_node.parent], region_mean, region_name.replace('_', ' '))
    plt.figure(0); plt.suptitle('True distributions with hierarchically estimated means'); plt.tight_layout();
    plt.figure(1); plt.suptitle('Posterior distributions with true means'); plt.tight_layout();

country_measurement_frame, province_measurement_frame, region_measurement_frame, country_param_frame, province_param_frame, region_param_frame = loadMeasurementsAndTruth(csv_dir, args.csv_file_prefix)

# getting sample means and variances
country_param_estimated = getEstimatedParamsForPartition(country_measurement_frame)
province_param_estimated = getEstimatedParamsForPartition(province_measurement_frame)
region_param_estimated = getEstimatedParamsForPartition(region_measurement_frame)
# reordering columns
country_param_estimated = country_param_estimated[country_param_frame.columns]
province_param_estimated = province_param_estimated[province_param_frame.columns]
region_param_estimated = region_param_estimated[region_param_frame.columns]
# calculating hierarchical covariances
province_param_estimated = getPartitionOmegas([country_0, country_1], country_measurement_frame, country_param_estimated, province_measurement_frame, province_param_estimated)
region_param_estimated = getPartitionOmegas(country_0.children + country_1.children, province_measurement_frame, province_param_estimated, region_measurement_frame, region_param_estimated)
# estimating hierarchical means
province_param_estimated = getHierarchicalMeanEstimate([country_0, country_1], country_param_estimated, province_param_estimated)
region_param_estimated = getHierarchicalMeanEstimate(country_0.children + country_1.children, province_param_estimated, region_param_estimated)

# plotting and measuring mean squared difference between estimated and sampled means
msd = np.power(region_param_estimated.loc['mean'] - region_param_estimated.loc['hier_mean'], 2).mean()
if args.plot_mean_accuracy:
    plot_suptitle, plot_filename = getMeanComparisonPlotSuptitleAndFilename(args.identity_scaling_value, args.use_random_matrix, msd)
    pm.plotSampleAccuracyAndEstimatedAccuracy(country_0.children + country_1.children, province_to_colour, region_param_frame, region_param_estimated, plot_suptitle)
    plt.savefig(os.path.join(image_dir, 'mean_comparisons', plot_filename))
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Mean squared difference between sample means and hierarchically estimated means: ' + str(msd))
if args.save_mean_accuracy:
    with open(os.path.join(results_csv_dir, 'scaling_accuracy.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.identity_scaling_value, msd])

# sampling from model and printing pairwise correlations
model_measurement_frame = sampleFromModel(country_0.children + country_1.children, province_param_estimated, region_param_estimated)
plt.figure()
ax = plt.subplot(121)
ax.matshow(region_measurement_frame.corr())
ax = plt.subplot(122)
ax.matshow(model_measurement_frame.corr())
plt.tight_layout()

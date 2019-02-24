"""
Recover the parameters used to create the hierarchical data by calculating the multiscale parameters,
and inference.
Useful line for editing:
    exec(open(os.path.join(os.environ['HOME'], '.pystartup').read())
"""
import os
import sys, argparse, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as ss
from sklearn.datasets import make_spd_matrix # for generating random covariance matrices
from decimal import Decimal # for printing numbers in scientific notation
from numpy.random import multivariate_normal

# command line arguments
parser = argparse.ArgumentParser(description='Recover the parameters used to create the hierarchical data by calculating the multiscale parameters, and inference.')
parser.add_argument('-i', '--use_random_matrix', help='Use scaled random matrix as the prior covariance matrix.', action='store_true', default=False)
parser.add_argument('-s', '--identity_scaling_value', help='The value to scale the identity matrix.', type=float, default=1.0)
parser.add_argument('-p', '--plot_mean_accuracy', help='Flag to make the mean accuracy plot.', action='store_true', default=False)
parser.add_argument('-a', '--save_metrics', help='Flag to save the scaling value and mean squared accuracy.', action='store_true', default=False)
parser.add_argument('-b', '--save_accuracy_plot', help='Flag to save the mean accuracy plot.', action='store_true')
parser.add_argument('-e', '--plot_correlation', help='Flag to make the correlation mean_comparison plot.', action='store_true')
parser.add_argument('-c', '--correlation_plot_filename', help='Where to save the correlation plot. If not entered, plot will not be saved.', type=str, default='')
parser.add_argument('-g', '--plot_variance_accuracy', help='Flag to plot the sample variances vs model variances.', action='store_true')
parser.add_argument('-j', '--variance_accuracy_plot_filename', help='Where to save the variance accuracy plot. If not entered, plot will not be saved.', type=str, default='')
parser.add_argument('-f', '--csv_file_prefix', help='Prefix to attach to csv file names.', type=str, default='')
parser.add_argument('-n', '--num_samples', help='Num samples to take from model.', type=int, default=1000)
parser.add_argument('-d', '--debug', help='Flag to enter debug mode.', action='store_true', default=False)
args = parser.parse_args()

pd.set_option('max_rows', 30); pd.set_option('max_columns', 0);
# defining directories
proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv', 'gaussian')
image_dir = os.path.join(proj_dir, 'images', 'gaussian')
results_csv_dir = os.path.join(proj_dir, 'csv', 'results')

# loading hierarchical tree structure and province colours
exec(open(os.path.join(py_dir, 'tree_and_colours.py')).read())

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

def getPriorCovarianceMatrix(use_random_matrix, num_children, identity_scale):
    if use_random_matrix:
        prior_covariance = make_spd_matrix(num_children, random_state=1)/100
    else:
        prior_covariance = identity_scale * np.identity(num_children)
    return prior_covariance

def getChildrenForParameterEstimation(node):
    return node.children[:-1]

def getNodeOmega(node, node_param_estimated, child_param_estimated):
    node_var = node_param_estimated.loc['var'][node.name]
    child_names = [c.name for c in getChildrenForParameterEstimation(node)]
    child_vars = child_param_estimated.loc['var'][child_names]
    node_nu = child_vars/node_var
    node_Omega = np.diag(child_vars) - np.outer(child_vars, child_vars)/node_var
    return child_names, node_nu, node_Omega

def getNodeExpectedOmega(node, node_measure_frame, node_param_estimated, child_measure_frame, child_param_estimated):
    n = child_measure_frame.shape[0] # num samples
    num_children = len(getChildrenForParameterEstimation(node))
    node_phi = getPriorCovarianceMatrix(args.use_random_matrix, num_children, args.identity_scaling_value)
    child_names, node_nu, node_Omega = getNodeOmega(node, node_param_estimated, child_param_estimated)
    post_covariance = np.linalg.inv(np.linalg.inv(node_phi) + n*np.linalg.inv(node_Omega))
    measurement_factor = child_measure_frame[child_names].mean() - node_measure_frame[node.name].mean()*node_nu
    omega_hat = n * np.matmul(post_covariance, np.matmul(np.linalg.inv(node_Omega), measurement_factor))
    return omega_hat, node_nu.values, np.diag(post_covariance)

def getHierarchicalMeanEstimatebyNode(node, node_param_estimated, child_param_estimated):
    child_names = [c.name for c in getChildrenForParameterEstimation(node)]
    node_mean = node_param_estimated.loc['mean'][node.name]
    child_nu = child_param_estimated.loc['nu'][child_names]
    child_omega = child_param_estimated.loc['omega'][child_names]
    return child_omega + node_mean*child_nu

def estimateEasyParams(estimated_parameter_frame, measure_frame):
    estimated_parameter_frame.loc['mean'] = measure_frame.mean()
    estimated_parameter_frame.loc['var'] = measure_frame.var()
    estimated_parameter_frame.loc['std'] = measure_frame.std()
    return estimated_parameter_frame

def estimateHierarchicalParamsFromParentNode(node, depth_to_measure_frame, depth_to_estimated_frame):
    child_names = [c.name for c in getChildrenForParameterEstimation(node)]
    omega_hat, nu, post_variance = getNodeExpectedOmega(node, depth_to_measure_frame[node.depth], depth_to_estimated_frame[node.depth], depth_to_measure_frame[node.depth+1], depth_to_estimated_frame[node.depth+1])
    depth_to_estimated_frame[node.depth+1].loc['omega'][child_names] = omega_hat
    depth_to_estimated_frame[node.depth+1].loc['nu'][child_names] = nu
    depth_to_estimated_frame[node.depth+1].loc['posterior_variance'][child_names] = post_variance
    hier_mean = getHierarchicalMeanEstimatebyNode(node, depth_to_estimated_frame[node.depth], depth_to_estimated_frame[node.depth+1])
    depth_to_estimated_frame[node.depth+1].loc['hier_mean'][child_names] = hier_mean
    excluded_child_name = node.children[-1].name
    node_hier_mean = depth_to_estimated_frame[node.depth].loc['hier_mean'][node.name]
    depth_to_estimated_frame[node.depth+1].loc['hier_mean'][excluded_child_name] = node_hier_mean - hier_mean.sum()
    return depth_to_estimated_frame

def getEstimatedParameterFrames(country_nodes, depth_to_measure_frame):
    parameters_to_estimate = ['mean', 'var', 'std', 'omega', 'nu', 'posterior_variance', 'hier_mean']
    country_param_estimated = pd.DataFrame(columns = country_measurement_frame.columns, index=parameters_to_estimate[0:3], dtype=float)
    country_param_estimated = estimateEasyParams(country_param_estimated, depth_to_measure_frame[0])
    country_param_estimated.loc['hier_mean'] = country_param_estimated.loc['mean']
    province_param_estimated = pd.DataFrame(columns = province_measurement_frame.columns, index=parameters_to_estimate, dtype=float)
    province_param_estimated = estimateEasyParams(province_param_estimated, depth_to_measure_frame[1])
    region_param_estimated = pd.DataFrame(columns = region_measurement_frame.columns, index=parameters_to_estimate, dtype=float)
    region_param_estimated = estimateEasyParams(region_param_estimated, depth_to_measure_frame[2])
    depth_to_estimated_frame = {0:country_param_estimated, 1:province_param_estimated, 2:region_param_estimated}
    for top_level_node in country_nodes:
        node_and_family = (top_level_node,) + top_level_node.children
        for node in node_and_family:
            depth_to_estimated_frame = estimateHierarchicalParamsFromParentNode(node, depth_to_measure_frame, depth_to_estimated_frame)
    return depth_to_estimated_frame

def sampleFromModelByNode(node, depth_to_model_measurement, depth_to_estimated_frame):
    child_names, node_nu, node_Omega = getNodeOmega(node, depth_to_estimated_frame[node.depth], depth_to_estimated_frame[node.depth+1])
    node_nu = np.array(node_nu)
    node_measures = np.array(depth_to_model_measurement[node.depth][node.name])
    child_omegas = np.array(depth_to_estimated_frame[node.depth+1].loc['omega'][child_names])
    means_for_sampling = np.array([node_nu*y + child_omegas for y in node_measures])
    samples = np.array([multivariate_normal(m, node_Omega)for m in means_for_sampling])
    depth_to_model_measurement[node.depth+1][child_names] = samples
    excluded_child_name = node.children[-1].name
    depth_to_model_measurement[node.depth+1][excluded_child_name] = node_measures - samples.sum(axis=1)
    return depth_to_model_measurement

def sampleFromModel(country_nodes, depth_to_estimated_frame, num_samples=1000):
    depth_to_model_measurement = {}
    for k, estimated_parameter_frame in depth_to_estimated_frame.items():
        depth_to_model_measurement[k] = pd.DataFrame(columns=estimated_parameter_frame.columns, dtype=float, index=range(num_samples))
    top_level_Omega = np.diag(depth_to_estimated_frame[0].loc['var'])
    top_level_samples = multivariate_normal(depth_to_estimated_frame[0].loc['hier_mean'], top_level_Omega, num_samples)
    depth_to_model_measurement[0][[node.name for node in country_nodes]] = top_level_samples
    for top_level_node in country_nodes:
        node_and_family = (top_level_node,) + top_level_node.children
        for node in node_and_family:
            depth_to_model_measurement = sampleFromModelByNode(node, depth_to_model_measurement, depth_to_estimated_frame)
    return depth_to_model_measurement

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

def plotAccuracyAndSave(scaling_value, is_random, est_hier_msd, parent_nodes, colour_dict, child_param_frame, child_param_estimated, image_dir=image_dir, is_save=True):
    plot_suptitle, plot_filename = getMeanComparisonPlotSuptitleAndFilename(scaling_value, is_random, est_hier_msd)
    save_name = os.path.join(image_dir, 'mean_comparisons', plot_filename)
    pm.plotSampleAccuracyAndEstimatedAccuracy(parent_nodes, colour_dict, child_param_frame, child_param_estimated, plot_suptitle)
    if is_save:
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show(block=False)
    return save_name

def saveMetricsToCsv(scaling_value, num_measures, est_hier_msd, est_true_msd, hier_true_msd, var_msd, results_csv_dir=results_csv_dir):
    scaling_accuracy_file = os.path.join(results_csv_dir, 'scaling_accuracy.csv')
    if not(os.path.isfile(scaling_accuracy_file)):
        with open(scaling_accuracy_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['identity_scaling_value', 'num_measures', 'est_hier_msd', 'est_true_msd', 'hier_true_msd', 'var_msd'])
    with open(scaling_accuracy_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([scaling_value, num_measures, est_hier_msd, est_true_msd, hier_true_msd, var_msd])

def plotPartitionCorrelation(ax, k, corr_matrix, min_corr, max_corr, title):
	cax = ax.matshow(corr_matrix, vmin=min_corr, vmax=max_corr)
	plt.xticks(fontsize='large');plt.yticks(fontsize='large');
	if k==0:
		plt.title(title)
		ax.set_xticks([])
	return cax

def plotHierarchyPairwiseCorrelation(depth_to_measure_frame, depth_to_model_measurement, image_dir=image_dir, file_name=''):
    fig = plt.figure(figsize=(5.75, 8.25))
    num_partitions = len(depth_to_measure_frame)
    for k in depth_to_measure_frame.keys():
        data_corr = depth_to_measure_frame[k].corr().values
        model_corr = depth_to_model_measurement[k].corr().values
        np.fill_diagonal(data_corr, 0.0); np.fill_diagonal(model_corr, 0.0);
        min_corr = np.min([data_corr.min(), model_corr.min()])
        max_corr = np.max([data_corr.max(), model_corr.max()])
        ax = plt.subplot(num_partitions, 2, 2*k+1)
        plotPartitionCorrelation(ax, k, data_corr, min_corr, max_corr, 'Data correlation')
        ax = plt.subplot(num_partitions, 2, 2*k+2)
        im = plotPartitionCorrelation(ax, k, model_corr, min_corr, max_corr, 'Model correlation')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, ax.get_position().y0, 0.05, ax.get_position().y1 - ax.get_position().y0])
        fig.colorbar(im, cax=cbar_ax)
    if file_name != '':
        save_name = os.path.join(image_dir, 'covariance_comparisons', file_name)
        plt.savefig(save_name)
    else:
        save_name = ''
        plt.show(block=False)
    return save_name

def getModelLikelihood(country_nodes, depth_to_estimated_frame, depth_to_measure_frame):
    num_measures = depth_to_measure_frame[0].shape[0]
    depth_to_likelihoods = {}
    for k, estimated_parameter_frame in depth_to_estimated_frame.items():
        depth_to_likelihoods[k] = pd.DataFrame(columns=estimated_parameter_frame.columns, dtype=float, index=range(num_measures))
    for top_level_node in country_nodes:
        node_and_family = (top_level_node,) + top_level_node.children
        for node in node_and_family:
            child_names, node_nu, node_Omega = getNodeOmega(node, depth_to_estimated_frame[node.depth], depth_to_estimated_frame[node.depth+1])
            child_depth = node.depth + 1
            child_measurements = depth_to_measure_frame[child_depth][child_names].values
            parent_measurements = depth_to_measure_frame[node.depth][node.name].values
            for i, measures in enumerate(zip(child_measurements, parent_measurements)):
                measure_dist = ss.multivariate_normal(node_nu*measures[1], node_Omega)
                depth_to_likelihoods[node.depth].loc[i][node.name] = measure_dist.logpdf(measures[0])
    return depth_to_likelihoods

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading measurements...')
country_measurement_frame, province_measurement_frame, region_measurement_frame, country_param_frame, province_param_frame, region_param_frame = loadMeasurementsAndTruth(csv_dir, args.csv_file_prefix)
depth_to_measure_frame = {0:country_measurement_frame, 1:province_measurement_frame, 2:region_measurement_frame}
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Estimating hierarchical parameters...')
depth_to_estimated_frame = getEstimatedParameterFrames([country_0, country_1], depth_to_measure_frame)
country_param_estimated, province_param_estimated, region_param_estimated = [depth_to_estimated_frame[i]for i in depth_to_estimated_frame.keys()]
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Measuring mean squared differences...')
est_hier_msd = np.sqrt(np.power(region_param_estimated.loc['mean'] - region_param_estimated.loc['hier_mean'], 2)).mean()
est_true_msd = np.sqrt(np.power(region_param_estimated.loc['mean'] - region_param_frame.loc['mean'], 2)).mean()
hier_true_msd = np.sqrt(np.power(region_param_frame.loc['mean'] - region_param_estimated.loc['hier_mean'], 2)).mean()
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Mean squared difference between sample means and hierarchically estimated means: ' + str(est_hier_msd))
if args.plot_mean_accuracy:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plotting sample and estimation accuracy...')
    plot_filename = plotAccuracyAndSave(args.identity_scaling_value, args.use_random_matrix, est_hier_msd, country_0.children + country_1.children, province_to_colour, region_param_frame, region_param_estimated, is_save=args.save_accuracy_plot)
    if args.save_accuracy_plot:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plot saved: ' + plot_filename)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling from model and plotting pairwise correlations...')
depth_to_model_measurement = sampleFromModel([country_0, country_1], depth_to_estimated_frame, num_samples=args.num_samples)
var_msd = np.power(depth_to_measure_frame[2].var() - depth_to_model_measurement[2].var(), 2).mean()
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Mean squared difference between sample variances and model variances: ' + str(var_msd))
if args.plot_correlation:
    plot_filename = plotHierarchyPairwiseCorrelation(depth_to_measure_frame, depth_to_model_measurement, file_name=args.correlation_plot_filename)
    if plot_filename != '':
        plt.savefig(plot_filename)
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plot saved: ' + plot_filename)
region_param_estimated.loc['model_var'] = depth_to_model_measurement[2].var()
if args.plot_variance_accuracy:
    fig = plt.figure()
    pm.plotTrueMeansVsEstimatedMeans(country_0.children + country_1.children, province_to_colour, region_param_estimated, region_param_estimated, param_name='var', estimated_name='model_var', title='Model variances vs Sample variances', ylabel='Model variance', xlabel='Sample variance')
    plt.legend()
    if args.variance_accuracy_plot_filename != '':
        plot_filename = os.path.join(image_dir, 'variance_comparisons', args.variance_accuracy_plot_filename)
        plt.savefig(plot_filename)
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Plot saved: ' + plot_filename)
    else:
        plt.show(block=False)
if args.save_metrics:
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving mean estimation accuracy...')
    saveMetricsToCsv(args.identity_scaling_value, depth_to_measure_frame[0].shape[0], est_hier_msd, est_true_msd, hier_true_msd, var_msd)
depth_to_likelihoods = getModelLikelihood([country_0, country_1], depth_to_estimated_frame, depth_to_measure_frame)

import numpy as np
import matplotlib.pyplot as plt

def plotSampleAccuracyAndEstimatedAccuracy(parent_nodes, colour_dict, child_param_frame, child_param_estimated, suptitle=''):
    fig = plt.figure(figsize=(8.0,4.25))
    plt.subplot(121)
    plotTrueMeansVsEstimatedMeans(parent_nodes, colour_dict, child_param_frame, child_param_estimated, estimated_name='mean', ylabel='Sample mean')
    plt.legend(fontsize='large')
    plt.subplot(122)
    plotTrueMeansVsEstimatedMeans(parent_nodes, colour_dict, child_param_frame, child_param_estimated, estimated_name='hier_mean', ylabel='Hierarchically estimated mean')
    plt.suptitle(suptitle)

def plotTrueMeansVsEstimatedMeans(parent_nodes, colour_dict, child_param_frame, child_param_estimated, param_name='mean', estimated_name='hier_mean', title='', ylabel='Estimated mean', xlabel='True mean'):
    for parent_node in parent_nodes:
        scatterWithColour(parent_node, colour_dict, child_param_frame, child_param_estimated, param_name, estimated_name)
    plt.xlabel(xlabel, fontsize='large'); plt.ylabel(ylabel, fontsize='large')
    larger_limit = np.max([np.ceil(plt.xlim()[1]), np.ceil(plt.ylim()[1])])
    plt.plot([0,larger_limit], [0,larger_limit], color='black', linestyle='--', alpha=0.3)
    plt.xlim(0,larger_limit); plt.ylim(0,larger_limit)
    plt.title(title, fontsize='large')

def scatterWithColour(parent_node, colour_dict, child_param_frame, child_param_estimated, param_name, estimated_name):
    child_names = [cn.name for cn in parent_node.children]
    colour = colour_dict[parent_node]
    plt.scatter(child_param_frame.loc[param_name][child_names], child_param_estimated.loc[estimated_name][child_names], color=colour, label=parent_node.name)

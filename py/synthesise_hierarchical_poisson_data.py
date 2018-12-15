"""
create some synthetic heirarchical data.
Useful line for editing:
    execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import pandas as pd
import numpy as np

pd.set_option('max_rows', 30)
np.random.seed(1798)

proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv', 'poisson')

# creating parameter frames
region_param_frame = pd.DataFrame()
region_means = np.linspace(0.1,12,24)
for i in range(24):
    region_param_frame['region_' + str(i)] = [region_means[i]]
region_param_frame.index = ['mean']

# loading tree structure
execfile(os.path.join(py_dir, 'tree_and_colours.py'))

region_nodes = province_0.children + province_1.children + province_2.children + province_3.children + province_4.children + province_5.children
province_names = ['province_' + str(i) for i in range(6)]
province_param_frame = pd.DataFrame(data=np.zeros((1,6)), index=['mean'], columns=province_names)
for region_node in region_nodes:
    region_name = region_node.name
    province_name = region_node.parent.name
    region_mean = region_param_frame.loc['mean'][region_name]
    province_param_frame.loc['mean'][province_name] += region_mean

province_nodes = country_0.children + country_1.children
country_names = ['country_0', 'country_1']
country_param_frame = pd.DataFrame(data=np.zeros((1,2)), index=['mean'], columns=country_names)
for province_node in province_nodes:
    province_name = province_node.name
    country_name = province_node.parent.name
    province_mean = province_param_frame.loc['mean'][province_name]
    country_param_frame.loc['mean'][country_name] += province_mean

# creating measurement frames
num_measurements = 1000

region_measurement_frame = pd.DataFrame(data=np.zeros((num_measurements,24)), columns=region_param_frame.columns)
for region_name in region_param_frame.columns:
    region_measurement_frame[region_name] = np.random.poisson(region_param_frame.loc['mean'][region_name], num_measurements)

province_measurement_frame = pd.DataFrame(data=np.zeros((num_measurements,6)), columns=province_names)
for province_node in province_nodes:
    province_name = province_node.name
    region_nodes = province_node.children
    region_names = [region_node.name for region_node in region_nodes]
    province_measurement_frame[province_name] = region_measurement_frame[region_names].sum(axis=1)

country_measurement_frame = pd.DataFrame(data=np.zeros((num_measurements,2)), columns=country_names)
for country_node in [country_0, country_1]:
    country_name = country_node.name
    province_nodes = country_node.children
    province_names = [province_node.name for province_node in province_nodes]
    country_measurement_frame[country_name] = province_measurement_frame[province_names].sum(axis=1)

region_param_frame.to_csv(os.path.join(csv_dir, 'region_param_frame.csv'), index_label='parameter')
province_param_frame.to_csv(os.path.join(csv_dir, 'province_param_frame.csv'), index_label='parameter')
country_param_frame.to_csv(os.path.join(csv_dir, 'country_param_frame.csv'), index_label='parameter')
region_measurement_frame.to_csv(os.path.join(csv_dir, 'region_measurement_frame.csv'), index_label='row_num')
province_measurement_frame.to_csv(os.path.join(csv_dir, 'province_measurement_frame.csv'), index_label='row_num')
country_measurement_frame.to_csv(os.path.join(csv_dir, 'country_measurement_frame.csv'), index_label='row_num')

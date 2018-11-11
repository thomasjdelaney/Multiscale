import os
execfile(os.path.join(os.environ['HOME'], '.pythonrc'))
import pandas as pd
import random as rd
from numpy.random import normal, seed

pd.set_option('max_rows', 30)
rd.seed(1798)
seed(1798)

proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
csv_dir = os.path.join(proj_dir, 'csv')

region_param_frame = pd.DataFrame()
for i in range(12):
    region_param_frame['region_' + str(i)] = [rd.uniform(i+1,i+2), rd.uniform((i+1)/4.0,(i+2)/4.0)]
region_param_frame.index = ['mu', 'sigma']

region_measurement_frame = pd.DataFrame()
for col in region_param_frame.columns:
    region_measurement_frame[col] = normal(region_param_frame.loc['mu', col], region_param_frame.loc['sigma', col], 1000)

province_param_frame = pd.DataFrame()
province_param_frame['province_0'] = [region_param_frame.loc['mu'][0:3].sum()]
province_param_frame['province_1'] = [region_param_frame.loc['mu'][3:7].sum()]
province_param_frame['province_2'] = [region_param_frame.loc['mu'][7:12].sum()]
province_param_frame.index = ['mu']

province_measurement_frame = pd.DataFrame()
province_measurement_frame['province_0'] = region_measurement_frame[region_measurement_frame.columns[0:3]].sum(axis=1)
province_measurement_frame['province_1'] = region_measurement_frame[region_measurement_frame.columns[3:7]].sum(axis=1)
province_measurement_frame['province_2'] = region_measurement_frame[region_measurement_frame.columns[7:12]].sum(axis=1)

country_param_frame = pd.DataFrame()
country_param_frame['country_0'] = [province_param_frame.loc['mu'][0:2].sum()]
country_param_frame['country_1'] = [province_param_frame.loc['mu'][2]]
country_param_frame.index = ['mu']

country_measurement_frame = pd.DataFrame()
country_measurement_frame['country_0'] = province_measurement_frame[province_param_frame.columns[0:2]].sum(axis=1)
country_measurement_frame['country_1'] = province_measurement_frame['province_2']

region_param_frame.to_csv(os.path.join(csv_dir, 'region_param_frame.csv'), index_label='parameter')
province_param_frame.to_csv(os.path.join(csv_dir, 'province_param_frame.csv'), index_label='parameter')
country_param_frame.to_csv(os.path.join(csv_dir, 'country_param_frame.csv'), index_label='parameter')
region_measurement_frame.to_csv(os.path.join(csv_dir, 'region_measurement_frame.csv'), index_label='row_num')
province_measurement_frame.to_csv(os.path.join(csv_dir, 'province_measurement_frame.csv'), index_label='row_num')
country_measurement_frame.to_csv(os.path.join(csv_dir, 'country_measurement_frame.csv'), index_label='row_num')

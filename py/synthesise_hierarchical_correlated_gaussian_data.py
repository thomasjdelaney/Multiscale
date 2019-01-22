"""
create some synthetic heirarchical data. This time, the regions are correlated.
Useful line for editing:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
"""
import os
execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import pandas as pd
import random as rd
import numpy as np
from numpy.random import multivariate_normal, seed
from sklearn.datasets import make_spd_matrix # for generating random covariance matrices

rd.seed(1798)
seed(1798)

proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
csv_dir = os.path.join(proj_dir, 'csv', 'gaussian')

def getRegionalMeans():
    return [rd.uniform(2,15) for i in range(0,24)]

def getIntraprovinceCorrelationCovarianceMatrix():
    covariance_0 = make_spd_matrix(3, random_state=1798)
    covariance_1 = make_spd_matrix(4, random_state=1798)
    covariance_2 = make_spd_matrix(5, random_state=1798)
    covariance_3 = make_spd_matrix(5, random_state=1916)
    covariance_4 = make_spd_matrix(4, random_state=1916)
    covariance_5 = make_spd_matrix(3, random_state=1916)
    intraprovince_correlation_covariance = np.block([[covariance_0, np.zeros([3,21])],
            [np.zeros([4,3]), covariance_1, np.zeros([4,17])],
            [np.zeros([5,7]), covariance_2, np.zeros([5,12])],
            [np.zeros([5,12]), covariance_3, np.zeros([5,7])],
            [np.zeros([4,17]), covariance_4, np.zeros([4,3])],
            [np.zeros([3,21]), covariance_5]])
    return intraprovince_correlation_covariance

def getRegionParamFrame(regional_means, covariance_matrix):
    region_param_frame = pd.DataFrame()
    for i in range(0,24):
        region_param_frame['region_'+str(i)] = [regional_means[i]]
    region_param_frame.index=['mean']
    variance_series = pd.Series(np.diag(covariance_matrix), name='var', index=region_param_frame.columns)
    region_param_frame = region_param_frame.append(variance_series)
    region_param_frame.loc['std'] = np.sqrt(region_param_frame.loc['var'])
    return region_param_frame

regional_means = getRegionalMeans()
covariance_matrix = getIntraprovinceCorrelationCovarianceMatrix()

region_param_frame = getRegionParamFrame(regional_means, covariance_matrix)
samples = multivariate_normal(regional_means, covariance_matrix, 1000)
region_measurement_frame = pd.DataFrame(samples, columns=region_param_frame.columns)

province_param_frame = pd.DataFrame()
province_param_frame['province_0'] = region_param_frame.loc[['mean', 'var']].T[0:3].sum()
province_param_frame['province_1'] = region_param_frame.loc[['mean', 'var']].T[3:7].sum()
province_param_frame['province_2'] = region_param_frame.loc[['mean', 'var']].T[7:12].sum()
province_param_frame['province_3'] = region_param_frame.loc[['mean', 'var']].T[12:17].sum()
province_param_frame['province_4'] = region_param_frame.loc[['mean', 'var']].T[17:21].sum()
province_param_frame['province_5'] = region_param_frame.loc[['mean', 'var']].T[21:24].sum()
province_stds = pd.Series(np.sqrt(province_param_frame.loc['var']), name='std')
province_param_frame = province_param_frame.append(province_stds)

province_measurement_frame = pd.DataFrame()
province_measurement_frame['province_0'] = region_measurement_frame[region_measurement_frame.columns[0:3]].sum(axis=1)
province_measurement_frame['province_1'] = region_measurement_frame[region_measurement_frame.columns[3:7]].sum(axis=1)
province_measurement_frame['province_2'] = region_measurement_frame[region_measurement_frame.columns[7:12]].sum(axis=1)
province_measurement_frame['province_3'] = region_measurement_frame[region_measurement_frame.columns[12:17]].sum(axis=1)
province_measurement_frame['province_4'] = region_measurement_frame[region_measurement_frame.columns[17:21]].sum(axis=1)
province_measurement_frame['province_5'] = region_measurement_frame[region_measurement_frame.columns[21:24]].sum(axis=1)

country_param_frame = pd.DataFrame()
country_param_frame['country_0'] = [province_param_frame.loc['mean'][0:3].sum()]
country_param_frame['country_1'] = [province_param_frame.loc['mean'][3:6].sum()]
country_param_frame.index = ['mean']

country_measurement_frame = pd.DataFrame()
country_measurement_frame['country_0'] = province_measurement_frame[province_param_frame.columns[0:3]].sum(axis=1)
country_measurement_frame['country_1'] = province_measurement_frame[province_param_frame.columns[3:6]].sum(axis=1)

region_param_frame.to_csv(os.path.join(csv_dir, 'corr_region_param_frame.csv'), index_label='parameter')
province_param_frame.to_csv(os.path.join(csv_dir, 'corr_province_param_frame.csv'), index_label='parameter')
country_param_frame.to_csv(os.path.join(csv_dir, 'corr_country_param_frame.csv'), index_label='parameter')
region_measurement_frame.to_csv(os.path.join(csv_dir, 'corr_region_measurement_frame.csv'), index_label='row_num')
province_measurement_frame.to_csv(os.path.join(csv_dir, 'corr_province_measurement_frame.csv'), index_label='row_num')
country_measurement_frame.to_csv(os.path.join(csv_dir, 'corr_country_measurement_frame.csv'), index_label='row_num')
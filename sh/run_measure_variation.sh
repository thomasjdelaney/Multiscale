#!/bin/bash

proj_dir=$HOME/Multiscale
py_dir=$proj_dir/py
res_dir=$proj_dir/csv/results

rm $res_dir/scaling_accuracy.csv

for i in {1..30}
do
  /usr/bin/python $py_dir/synthesise_hierarchical_correlated_gaussian_data.py --covariance_type independent --save_prefix indy_ --num_samples "$i"0
  /usr/bin/python $py_dir/recover_synthetic_gaussian_parameters.py --save_metrics --csv_file_prefix indy_
done

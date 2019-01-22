#!/bin/bash

proj_dir=$HOME/Multiscale
py_dir=$proj_dir/py
res_dir=$proj_dir/csv/results

rm $res_dir/scaling_accuracy.csv

for scaling in 0.0001 0.001 0.01 0.05 0.1 0.25 0.5 0.75 1.0 1.5
do
  /usr/bin/python $py_dir/recover_synthetic_gaussian_parameters.py --identity_scaling_value $scaling --save_mean_accuracy
done

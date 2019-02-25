"""
For plotting two or more marginal likelihoods over the number of measurements.
"""
import os
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

parser = argparse.ArgumentParser(description='For plotting two or more marginal likelihoods over the number of measurements.')
parser.add_argument('-f', '--files', help='The list of files to plot.', type=str, nargs='+', default=['false_tree_ml.npy', 'true_tree_ml.npy'])
parser.add_argument('-l', '--labels', help='The labels for each file.', type=str, nargs='*', default=[''])
args = parser.parse_args()

proj_dir = os.path.join(os.environ['HOME'], 'Multiscale')
npy_dir = os.path.join(proj_dir, 'npy')

if args.labels == ['']:
    labels = args.files
else:
    if len(args.labels) != len(args.files):
        sys.exit(dt.datetime.now().isoformat() + ' ERROR: ' + 'Number of labels != number of files')
    else:
        labels=args.labels

for i,f in enumerate(args.files):
    ml = np.load(os.path.join(npy_dir, 'marginal_likelihoods', f)).cumsum()
    plt.plot(ml, label=labels[i])
plt.xlabel('Number of Measurements', fontsize='large')
plt.ylabel('Marginal Likelihood (nat)', fontsize='large')
plt.legend(fontsize='large')
plt.tight_layout()

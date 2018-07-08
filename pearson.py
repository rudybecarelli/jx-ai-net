import argparse
import seaborn

from matplotlib import pyplot
import numpy as np
import os
from pandas import read_csv


# Read args
parser = argparse.ArgumentParser(description='Produces the Pearson''s correlation matrix for x.csv' )
parser.add_argument('training_data_folder', help='Training data folder')
training_data_folder = parser.parse_args().training_data_folder


# Load x
x = read_csv(os.path.join(training_data_folder, 'x_train.csv'), header=None, index_col=False).values.astype('float32')

# Compute the Pearson's matrix
cm = np.absolute(np.corrcoef(x.transpose()))

np.savetxt(os.path.join(training_data_folder, 'pearson.csv'), cm, delimiter=",", fmt='%.2f')

seaborn.heatmap(cm, square=True)

pyplot.savefig(os.path.join(training_data_folder, 'pearson.png'))

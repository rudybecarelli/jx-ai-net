from matplotlib import pyplot
from pandas import read_csv
import argparse
import ConfigParser
import os
import numpy as np
from keras.models import load_model

# Parameter defaults
defaults = {
    'name': 'default_jxai',
    'root': '.',
    'log_to_file': True
}

# Load configuration file
parser = argparse.ArgumentParser(description='Train the jx-ai net')
parser.add_argument('conf_file_path', help='The path to the conf file')
conf_file_path = parser.parse_args().conf_file_path

# Override defaults if the configuration file exists
config_parser = ConfigParser.ConfigParser(defaults=defaults)
if os.path.isfile(conf_file_path):
    config_parser.read(conf_file_path)
base_section = config_parser.sections()[0]

name = config_parser.get(base_section, 'name')
file_root = config_parser.get(base_section, 'root')
log_to_file = config_parser.getboolean(base_section, 'log_to_file')

# Load dataset
x = read_csv(os.path.join(file_root, 'x.csv'), header=None, index_col=False).values.astype('float32')
y = read_csv(os.path.join(file_root, 'y.csv'), header=None, index_col=False).values.astype('float32')

# Reshape input to be 3D [samples, timesteps, features]
x = x.reshape((x.shape[0], 1, x.shape[1]))

# Save the model
model = load_model(os.path.join(file_root, name + '.h5'))

# Make a prediction on the whole y
y_hat = model.predict(x)

# Save y_hat
np.savetxt(os.path.join(file_root, 'y_hat.csv'), y_hat, delimiter=",")

# Log to file
if log_to_file:

    # Plot y-yhat
    pyplot.figure()
    pyplot.subplot(211)
    pyplot.title('y')
    pyplot.plot(y)
    pyplot.subplot(212)
    pyplot.title('y_hat')
    pyplot.plot(y_hat)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(file_root, name + '_y_yhat.png'))

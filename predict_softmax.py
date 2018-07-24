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
    'create_graph': 'False'
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
create_graph = config_parser.getboolean(base_section, 'create_graph')

# Load dataset
x = np.load(os.path.join(file_root, 'x_test.npy'))
y = np.load(os.path.join(file_root, 'y_test.npy'))

# Save the model
model = load_model(os.path.join(file_root, name + '.h5'))

# Make a prediction on the whole y
y_hat = model.predict(x)

# Hardmax the softmax variable
def to_hardmax(x):

    one_position = np.argmax(x)

    x_hardmax = np.zeros(x.shape, dtype=np.int32)

    x_hardmax[one_position] = 1

    return x_hardmax

y_hat = np.apply_along_axis( to_hardmax, axis=1, arr=y_hat )

a = np.absolute(np.subtract(y, y_hat))

errors = np.sum(np.sum(np.absolute(np.subtract(y, y_hat))))

print(errors / (2 * y.shape[0]))

# Save y_hat
np.save(os.path.join(file_root, 'y_hat'), y_hat)

# Log to file
if create_graph:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

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

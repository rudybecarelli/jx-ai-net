import matplotlib
from math import floor
from pandas import read_csv
import keras.callbacks as callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import argparse
import ConfigParser
import os


# Parameter defaults
defaults = {
    'name': 'default_jxai',
    'root': '.',
    'nodes': 100,
    'split': 0.8,
    'epochs': 300,
    'batch': 20,
    'crossvalid': False,
    'verbosity': 0,
    'log_to_file': False,
    'online_plot': False
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
lstm_nodes = config_parser.getint(base_section, 'nodes')
train_ratio = config_parser.getfloat(base_section, 'split')
epochs = config_parser.getint(base_section, 'epochs')
batch_size = config_parser.getint(base_section, 'batch')
crossvalidation = config_parser.getboolean(base_section, 'crossvalid')
verbosity = config_parser.getint(base_section, 'verbosity')
log_to_file = config_parser.getboolean(base_section, 'log_to_file')
online_plot = config_parser.getboolean(base_section, 'online_plot')

if not online_plot:
    matplotlib.use('Agg')

from matplotlib import pyplot

# Plotting class
class Plotter(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

        if online_plot:
            self.plot_curves()

    def plot_curves(self):
        pyplot.clf()
        pyplot.plot(self.loss, label="loss")
        pyplot.plot(self.val_loss, label="val_loss")
        pyplot.plot(self.acc, label="acc")
        pyplot.plot(self.val_acc, label="val_acc")
        pyplot.legend()
        text_log = 'loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' % \
                   (self.loss[-1], self.val_loss[-1], self.acc[-1], self.val_acc[-1])
        pyplot.xlabel(text_log)
        pyplot.show(block=False)
        pyplot.pause(0.01)

# Create the plotter
plotter = Plotter()

# Load dataset
x = read_csv(os.path.join(file_root, 'x.csv'), header=None, index_col=False).values.astype('float32')
y = read_csv(os.path.join(file_root, 'y.csv'), header=None, index_col=False).values.astype('float32')

# Split into train and test sets
train_length = int(floor(train_ratio * x.shape[0]))
train_x, train_y = x[:train_length, :], y[:train_length, :]
test_x, test_y = x[train_length:, :], y[train_length:, :]

# Reshape input to be 3D [samples, timesteps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# Design network
model = Sequential()
model.add(LSTM(lstm_nodes, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_y.shape[1]))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

# Fit network
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=verbosity,
                    shuffle=False, callbacks=[plotter])

# Save the model
model.save(os.path.join(file_root, name + '.h5'))

if log_to_file:
    plotter.plot_curves()
    pyplot.savefig(os.path.join(file_root, name + '_logs.png'))

if online_plot:
    pyplot.ioff()
    pyplot.show()

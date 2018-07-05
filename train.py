from math import floor
from pandas import read_csv
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
    'create_graph': False
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
create_graph = config_parser.getboolean(base_section, 'create_graph')

if create_graph:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

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
                    shuffle=False)

# Save the model
model.save(os.path.join(file_root, name + '.h5'))

if create_graph:
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    pyplot.figure()
    pyplot.plot(loss, label="loss")
    pyplot.plot(val_loss, label="val_loss")
    pyplot.plot(acc, label="acc")
    pyplot.plot(val_acc, label="val_acc")
    pyplot.legend()
    text_log = 'loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' % \
               (loss[-1], val_loss[-1], acc[-1], val_acc[-1])
    pyplot.xlabel(text_log)
    pyplot.savefig(os.path.join(file_root, name + '_logs.png'))

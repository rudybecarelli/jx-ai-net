from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import argparse
import ConfigParser
import os
import numpy as np

# Parameter defaults
defaults = {
    'name': 'default_jxai',
    'root': '.',
    'nodes': '100',
    'split': '0.8',
    'epochs': '300',
    'batch': '20',
    'dropout': '0.2',
    'crossvalid': 'False',
    'verbosity': '0',
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
lstm_nodes = config_parser.getint(base_section, 'nodes')
train_ratio = config_parser.getfloat(base_section, 'split')
epochs = config_parser.getint(base_section, 'epochs')
batch_size = config_parser.getint(base_section, 'batch')
dropout = config_parser.getfloat(base_section, 'dropout')
crossvalidation = config_parser.getboolean(base_section, 'crossvalid')
verbosity = config_parser.getint(base_section, 'verbosity')
create_graph = config_parser.getboolean(base_section, 'create_graph')

# Load dataset
x_train = np.load(os.path.join(file_root, 'x_train.npy'))
y_train = np.load(os.path.join(file_root, 'y_train.npy'))
x_test = np.load(os.path.join(file_root, 'x_test.npy'))
y_test = np.load(os.path.join(file_root, 'y_test.npy'))

# Design network
model = Sequential()
model.add(LSTM(lstm_nodes, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(lstm_nodes))
model.add(Dropout(dropout))
model.add(Dense(lstm_nodes))
model.add(Dropout(dropout))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit network
history = model.fit(x_train, y_train,
                    epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                    verbose=verbosity, shuffle=False)

# Save the model
model.save(os.path.join(file_root, name + '.h5'))

if create_graph:
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot

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
    #pyplot.ylim((0, 1))
    pyplot.savefig(os.path.join(file_root, name + '_graphs.png'))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
import argparse
import ConfigParser
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Parameter defaults

defaults = {
    'name': 'default_jxai',
    'root': '.',
    'nodes': '100',
    'k_fold_splits': '20',
    'epochs': '300',
    'batch': '20',
    'dropout': '0.2',
    'crossvalid': 'False',
    'verbosity': '0',
    'create_graph': 'False'
}

# Arrays for the graph

loss = []

val_loss = []

acc = []

val_acc = []

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

k_fold_splits = config_parser.getint(base_section, 'k_fold_splits')

epochs = config_parser.getint(base_section, 'epochs')

batch_size = config_parser.getint(base_section, 'batch')

dropout = config_parser.getfloat(base_section, 'dropout')

verbosity = config_parser.getint(base_section, 'verbosity')

create_graph = config_parser.getboolean(base_section, 'create_graph')

# Load dataset

x = np.load(os.path.join(file_root, 'x_train.npy'))

y = np.load(os.path.join(file_root, 'y_train.npy'))

# Random permutation

permuted_indexes = np.random.permutation(x.shape[0])

x = x[permuted_indexes]

y = y[permuted_indexes]

# Model path

model_path = os.path.join(file_root, name + '.h5')

# Design network

model = Sequential()

model.add(LSTM(lstm_nodes, input_shape=(x.shape[1], x.shape[2])))

model.add(Dense(lstm_nodes))

#model.add(Dropout(dropout))

model.add(Dense(lstm_nodes))

#model.add(Dropout(dropout))

model.add(Dense(y.shape[1], activation='softmax'))

adam = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Load the model weights

if os.path.isfile(model_path):

    model.load_weights(model_path)

# Instantiate the cross validator

skf = StratifiedKFold(n_splits=k_fold_splits)

# Loop through the indices the split() method returns

for index, (train_indices, val_indices) in enumerate(skf.split(x[:, 0, 0], y[:, 0])):

    # Generate batches from indices

    x_train, x_test = x[train_indices], x[val_indices]

    y_train, y_test = y[train_indices], y[val_indices]

    # Fit network

    history = model.fit(x_train, y_train,
                        epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=verbosity, shuffle=True)

    # Update graph data

    loss = np.hstack((loss, history.history["loss"]))

    val_loss = np.hstack((val_loss, history.history["val_loss"]))

    acc = np.hstack((acc, history.history["acc"]))

    val_acc = np.hstack((val_acc, history.history["val_acc"]))

# Save the model

model.save(model_path)

# Plot the graph

if create_graph:

    import matplotlib

    matplotlib.use('Agg')

    from matplotlib import pyplot

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

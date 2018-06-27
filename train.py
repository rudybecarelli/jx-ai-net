from math import sqrt
from math import floor
from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import argparse
import ConfigParser
import os
import numpy as np

# Parameter defaults
name = 'default_jxai'
file_root = '.'
lstm_nodes = 50
train_ratio = 0.5
epochs = 50
batch_size = 100
crossvalidation = False
training_verbosity = 0
log_to_file = True
log_to_std_out = True

# Load configuration file
parser = argparse.ArgumentParser(description='Train the jx-ai net')
parser.add_argument('conf_file_path', help='The path to the conf file')
conf_file_path = parser.parse_args().conf_file_path

# Override defaults if the configuration file exists
if os.path.isfile(conf_file_path):
    parameters = {}
    config_parser = ConfigParser.ConfigParser()
    config_parser.read(conf_file_path)
    sections = config_parser.sections()
    options = config_parser.options(sections[0])
    for option in options:
        parameters[option] = config_parser.get(sections[0], option)

    # Override parameter defaults
    if 'name' in parameters:
        name = parameters['name']

    if 'root' in parameters:
        file_root = parameters['root']

    if 'nodes' in parameters:
        lstm_nodes = int(parameters['nodes'])

    if 'split' in parameters:
        train_ratio = float(parameters['split'])

    if 'epochs' in parameters:
        epochs = int(parameters['epochs'])

    if 'batch' in parameters:
        batch_size = int(parameters['batch'])

    if 'training_verbosity' in parameters:
        training_verbosity = int(parameters['training_verbosity'])

    if 'crossvalid' in parameters:
        if parameters['crossvalid'] == 'off':
            crossvalidation = False
        else:
            crossvalidation = True

    if 'log_to_file' in parameters:
        if parameters['log_to_file'] == 'off':
            log_to_file = False
        else:
            log_to_file = True

    if 'log_to_std_out' in parameters:
        if parameters['log_to_std_out'] == 'off':
            log_to_std_out = False
        else:
            log_to_std_out = True

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
model.compile(loss='mae', optimizer='adam')

# Fit network
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=training_verbosity,
                    shuffle=False)

# Save the model
model.save(os.path.join(file_root, name + '.h5'))

# Make a prediction on the whole y and on the last sample
y_hat = model.predict(test_x)
y_hat_1 = y_hat[None, -1]
y_hat_last = model.predict(test_x[None, -1])

# Compute the validation set RMSE and last sample predict
rmse = sqrt(mean_squared_error(test_y, y_hat))
last_x_predict = np.array2string(y_hat_last, precision=16, separator=',').replace('[', '').replace(']', '').replace(' ', '')

# Log to std_out
if log_to_std_out:

    # Write tuning information
    print(str(rmse))
    print(last_x_predict)

# Log to file
if log_to_file:

    # Plot history
    pyplot.figure()
    pyplot.plot(history.history['loss'], label='train loss')
    pyplot.plot(history.history['val_loss'], label='validation loss')
    pyplot.legend()
    pyplot.savefig(os.path.join(file_root, name + '_losses.png'))

    # Write tuning information
    with open(os.path.join(file_root, name + '.log'), "w") as log_file:
        log_file.write(str(rmse) + '\n')
        log_file.write(last_x_predict)

    # Plot y-yhat
    pyplot.figure()
    pyplot.subplot(211)
    pyplot.title('y')
    pyplot.plot(test_y)
    pyplot.subplot(212)
    pyplot.title('y_hat')
    pyplot.plot(y_hat)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(file_root, name + '_y_yhat.png'))

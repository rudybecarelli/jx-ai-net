import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import math


train_ratio = 0.8

# Read the original file and create the onehot encoding

x_tommy_denorm = pd.read_csv('x_tommy_denorm.csv', header=None)

keys = x_tommy_denorm.keys()

x_local = None

y_local = None

plant_indexes = None

for key in keys:

    x_column = x_tommy_denorm[key]

    if key == 0:

        plant_indexes = x_column

    else:

        # Integer encode

        label_encoder = LabelEncoder()

        integer_encoded = label_encoder.fit_transform(x_column)

        # Binary encode

        onehot_encoder = OneHotEncoder(sparse=False)

        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        if x_local is None:

            x_local = onehot_encoded

        else:

            x_local = np.hstack((x_local, onehot_encoded))

        if key == 5:

            y_local = onehot_encoded

# Create the cube dictionary

cube_dict_x = {}

cube_dict_y = {}

for index, value in plant_indexes.iteritems():

    x_row = np.transpose(np.expand_dims(x_local[index, :], axis=1))

    y_row = np.transpose(np.expand_dims(y_local[index], axis=1))

    if value in cube_dict_x:

        time_series = cube_dict_x[value]

        time_series = np.vstack((time_series, x_row))

        cube_dict_x[value] = time_series

        #

        time_series = cube_dict_y[value]

        time_series = np.vstack((time_series, y_row))

        cube_dict_y[value] = time_series

    else:

        cube_dict_x[value] = x_row

        cube_dict_y[value] = y_row

# Look for the max size to pad to

maxes = np.array([])

for key, value in cube_dict_x.iteritems():

    maxes = np.hstack((maxes, value.shape[0]))

max_for_padding = np.amax(maxes)

# Pad the cubes

for key, value in cube_dict_x.iteritems():

    cube_slice = cube_dict_x[key]

    pad_size = int(max_for_padding - cube_slice.shape[0])

    cube_slice = np.pad(cube_slice, ((0, pad_size), (0, 0)), 'constant')

    cube_dict_x[key] = cube_slice

    #

    cube_slice = cube_dict_y[key]

    cube_slice = np.pad(cube_slice, ((0, pad_size), (0, 0)), 'constant')

    cube_dict_y[key] = cube_slice

# Transform the cube_dicts to np cubes

x = None

y = None

for key, value in cube_dict_x.iteritems():

    cube_slice = cube_dict_x[key]

    cube_slice = np.expand_dims(cube_slice, axis=2)

    cube_slice = np.transpose(cube_slice, (2, 0, 1))

    if x is None:

        x = cube_slice

    else:

        x = np.append(x, cube_slice, axis=0)

    #

    cube_slice = cube_dict_y[key]

    cube_slice = np.expand_dims(cube_slice, axis=2)

    cube_slice = np.transpose(cube_slice, (2, 0, 1))

    if y is None:

        y = cube_slice

    else:

        y = np.append(y, cube_slice, axis=0)

x = np.delete(x, -1, axis=1).astype(np.int32)

y = np.delete(y, 0, axis=1).astype(np.int32)

train_limit = int(math.floor(x.shape[0] * train_ratio))

x_train = x[:train_limit, :, :]

x_test = x[train_limit:, :, :]

y_train = y[:train_limit, :, :]

y_test = y[train_limit:, :, :]

np.save("x_train", x_train)

np.save("x_test", x_test)

np.save("y_train", y_train)

np.save("y_test", y_test)

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

train_ratio = 0.95

sequence_depth = 3

# Read the original file

x_denorm = pd.read_csv('x_tommy_denorm.csv', header=None)

# Drop invalid data

x_denorm = x_denorm.drop([2, 5, 10], axis=1)

# Label and onehot encoding

x_scaled = x_denorm.copy()

y_onehot = None

for key in x_denorm.keys():

    if key != 0:

        # Label encoding

        x_encoded_column = LabelEncoder().fit_transform(x_denorm[key]).reshape(-1, 1)

        x_scaled[key] = MinMaxScaler().fit_transform(x_encoded_column.astype(np.float64))

        if key == 3:

            # Onehot encoding

            y_onehot = OneHotEncoder(sparse=False).fit_transform(x_encoded_column)

# Extract the sequences

ids = set(x_scaled[0])

x = np.empty((0, sequence_depth, len(x_scaled.columns) - 1))

y = np.empty((0, y_onehot.shape[1]))

for id in ids:

    sequence = x_scaled[x_scaled[0] == id]

    if len(sequence.index) >= sequence_depth + 1:

        sequence = sequence.drop([0], axis=1)

        sequence_indexes = sequence.index.values

        for row_index in np.arange(0, len(sequence_indexes) - sequence_depth):

            x_slice = np.zeros((1, sequence_depth, len(sequence.columns)))

            for i in np.arange(0, sequence_depth):

                x_slice[0, i, :] = sequence.iloc[row_index + i].values

            x = np.vstack((x, x_slice))

            y_slice = np.zeros((1, y_onehot.shape[1]))

            y_slice[0, :] = y_onehot[sequence_indexes[row_index + sequence_depth]]

            y = np.vstack((y, y_slice))

# Save the variables

train_limit = int(math.floor(x.shape[0] * train_ratio))

x_train = x[:train_limit, :, :]

x_test = x[train_limit:, :, :]

y_train = y[:train_limit, :]

y_test = y[train_limit:, :]

np.save("x_train", x_train)

np.save("x_test", x_test)

np.save("y_train", y_train)

np.save("y_test", y_test)

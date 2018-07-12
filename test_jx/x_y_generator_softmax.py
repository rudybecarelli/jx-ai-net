import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import math


train_ratio = 0.8

#

x_tommy_denorm = pd.read_csv('x_tommy_denorm.csv', header=None)
keys = x_tommy_denorm.keys()

x_local = None

y_local = None

for key in keys:

    x_column = x_tommy_denorm[key]

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(x_column)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    if x_local is None:
        x_local = onehot_encoded
    else:
        x_local = np.hstack((x_local, onehot_encoded))

    if key == 5:
        y_local = onehot_encoded

x = np.delete(x_local, -1, axis=0).astype(np.int32)

y = np.delete(y_local, 0, axis=0).astype(np.int32)

train_limit = int(math.floor(x.shape[0] * train_ratio))

x_train = x[:train_limit]

x_test = x[train_limit:]

y_train = y[:train_limit]

y_test = y[train_limit:]

np.savetxt("x_train.csv", x_train, delimiter=",")

np.savetxt("x_test.csv", x_test, delimiter=",")

np.savetxt("y_train.csv", y_train, delimiter=",")

np.savetxt("y_test.csv", y_test, delimiter=",")

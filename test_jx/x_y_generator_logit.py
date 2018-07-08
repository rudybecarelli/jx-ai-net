import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import math


train_ratio = 0.8

#

x_tommy_denorm = pd.read_csv('x_tommy_denorm.csv', header=None);
keys = x_tommy_denorm.keys()

x_local = None

y_local = None

for key in keys:

    x_column = x_tommy_denorm[key]

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(x_column)

    normalized_encoded = integer_encoded - np.min(integer_encoded)

    a = np.max(integer_encoded)

    normalized_encoded = np.true_divide(normalized_encoded, np.max(integer_encoded))

    if x_local is None:
        x_local = np.expand_dims(normalized_encoded, 1)
    else:
        x_local = np.hstack((x_local, np.expand_dims(normalized_encoded, 1)))

    if key in (6,7,8):
        if y_local is None:
            y_local = np.expand_dims(normalized_encoded, 1)
        else:
            y_local = np.hstack((y_local, np.expand_dims(normalized_encoded, 1)))

x = np.delete(x_local, -1, axis=0).astype(np.int32)

y = np.delete(y_local, 0, axis=0).astype(np.int32)

train_limit = int(math.floor(x.shape[0] * train_ratio));

x_train = x[:train_limit]

x_test = x[train_limit:]

y_train = y[:train_limit]

y_test = y[train_limit:]

np.savetxt("x_train.csv", x_train, delimiter=",")

np.savetxt("x_test.csv", x_test, delimiter=",")

np.savetxt("y_train.csv", y_train, delimiter=",")

np.savetxt("y_test.csv", y_test, delimiter=",")

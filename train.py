from math import sqrt
from math import floor
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Parameters
train_ratio = 0.2 # 0.2
lstm_nodes = 50 # 50
epochs = 100 # 100
batch_size = 72 # 72

# Load dataset
x = read_csv('x.csv', header=None, index_col=False).values.astype('float32')
y = read_csv('y4.csv', header=None, index_col=False).values.astype('float32')

# Normalize features
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_x = x_scaler.fit_transform(x)
scaled_y = y_scaler.fit_transform(y)

# Split into train and test sets
train_length = int(floor(train_ratio * x.shape[0]))
train_x, train_y = scaled_x[:train_length, :], scaled_y[:train_length, :]
test_x, test_y = scaled_x[train_length:, :], scaled_y[train_length:, :]

# Reshape input to be 3D [samples, timesteps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# Design network
model = Sequential()
model.add(LSTM(lstm_nodes, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_y.shape[1]))
model.compile(loss='mae', optimizer='adam')

# Fit network
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2,
                    shuffle=False)

# Save the model
model.save('model.h5')

# Plot history
pyplot.figure()
pyplot.plot(history.history['loss'], label='train loss')
pyplot.plot(history.history['val_loss'], label='validation loss')
pyplot.legend()
pyplot.show(block=False)

# Make a prediction
yhat = model.predict(test_x)

# Denormalize features
inv_yhat = y_scaler.inverse_transform(yhat)
inv_y = y_scaler.inverse_transform(test_y)

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# plot y-yhat
pyplot.figure()
pyplot.subplot(211)
pyplot.title('y')
pyplot.plot(inv_y)
pyplot.subplot(212)
pyplot.title('yhat')
pyplot.plot(inv_yhat)
pyplot.show()

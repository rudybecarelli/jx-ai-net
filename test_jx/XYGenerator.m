x_cols = 1:9;

y_cols = 1:9;

train_ratio = 0.8;

%

x_default = csvread('x_default.csv');

x = x_default(1:end-1, x_cols);

y = x_default(2:end, y_cols);

train_limit = floor(size(x, 1) * train_ratio);

csvwrite('x_train.csv', x(1:train_limit, :));

csvwrite('y_train.csv', y(1:train_limit, :));

csvwrite('x_test.csv', x(train_limit + 1:end, :));

csvwrite('y_test.csv', y(train_limit + 1:end, :));

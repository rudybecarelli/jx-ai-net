x_in = csvread('x_denorm.csv');

y_in = csvread('y_denorm.csv');

cut_factor = 0.8;

dataset_limit = 900;

a = x_in - min(x_in);

x = a ./ max(a);

a = y_in - min(y_in);

y = a ./ max(a);

cut = floor( dataset_limit * cut_factor);

x_train = x(1:cut, :);

y_train = y(1:cut, :);

x_test = x((cut + 1):dataset_limit, :);

y_test = y((cut + 1):dataset_limit, :);

csvwrite('x_train.csv', x_train);

csvwrite('y_train.csv', y_train);

csvwrite('x_test.csv', x_test);

csvwrite('y_test.csv', y_test);

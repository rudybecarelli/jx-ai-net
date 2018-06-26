x_in = csvread('x_denorm.csv');

y_in = csvread('y_denorm.csv');

a = x_in - min(x_in);

x = a ./ max(a);

a = y_in - min(y_in);

y = a ./ max(a);

csvwrite('x.csv', x);

csvwrite('y.csv', y);

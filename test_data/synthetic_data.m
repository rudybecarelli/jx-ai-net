[X, Y] = meshgrid(1:8, 1:901);

w1 = 10;

w2 = 10;

x = sin((w1 * X) + (w2 * Y));

x = x - min(min(x));

x = x / max(max(x));

surf(X, Y, x);

y = x(2:end, (end - 3):end);

x(end, :) = [];

csvwrite('x.csv', x);

csvwrite('y.csv', y);
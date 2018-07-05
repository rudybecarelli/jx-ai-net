x = round(rand(901, 8));

surf(x);

y = x(2:end, (end - 3):end);

x(end, :) = [];

csvwrite('x.csv', x);

csvwrite('y.csv', y);
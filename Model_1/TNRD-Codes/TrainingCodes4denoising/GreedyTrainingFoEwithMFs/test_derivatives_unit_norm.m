clear all; clc;
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 16;
basis = gen_dct2(filter_size);
basis = basis(:,2:filter_num+1);

x0 = (rand(filter_num,1) - 0.5)*2;
f = basis*x0/norm(x0);
norm(f)

eps = 1e-8;
Jacob = zeros(length(x0),length(f));
for row = 1:length(x0)
    x = x0;
    x(row) = x(row) + eps;
    fp = basis*x/norm(x);

    x = x0;
    x(row) = x(row) - eps;
    fm = basis*x/norm(x);

    Jacob(row,:) = (fp - fm)/(2*eps);
end
deriMtx = 1/norm(x)*(eye(length(x)) - x*x'/norm(x)^2)*basis';
delta = Jacob - deriMtx;




function [x0, mfs] = Equal_Initialization(KernelPara, stage)
load w0_63_means.mat;
% load w0_125_means.mat;
mfs.means = means;
mfs.precision = precision;
mfs.NumW = length(means);
step = 0.2;
delta = 10;
D = -delta+means(1):step:means(end)+delta;
D_mu = bsxfun(@minus, D, means(:));
mfs.step = step;
mfs.D = D;
mfs.D_mu = D_mu;
mfs.offsetD = D(1);
mfs.nD = numel(D);
mfs.G = exp(-0.5*mfs.precision*D_mu.^2);


filter_size = KernelPara.fsz;
filtN = KernelPara.filtN;
m = filter_size^2 - 1;
w = repmat(w,[1,filtN]);
cof_beta = eye(m,m);
x0 = zeros(length(cof_beta(:)) + 1 + filtN*mfs.NumW, stage);
theta = [10 5 ones(1,stage-2)];
p = [log(1) log(0.1)*ones(1,stage-1)];
% beta = [log(1) log(0.1)*ones(1,stage-1)];

for i=1:stage
    x0(:,i) = [cof_beta(:); p(i); w(:)*theta(i)];
end
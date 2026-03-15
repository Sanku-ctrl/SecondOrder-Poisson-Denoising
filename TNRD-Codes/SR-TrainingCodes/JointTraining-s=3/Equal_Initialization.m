function [x0, mfs] = Equal_Initialization(KernelPara, run_stage)
% rp = [1:20, 30:10:150];
% lp = [-150:10:-20, -19:-1];
% means = [lp, 0, rp]';
% precision = [0.01/16*ones(1,5), 0.01*ones(1,9), 2*ones(1,39), 0.01*ones(1,9), 0.01/16*ones(1,5)]';
% load EqualInitialW.mat;
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
% load training_5x5_400_180x180_s=3.mat;
load greedy_training_7x7_400_180x180_s=3.mat;
x0 = cof(:,1:run_stage);
% filter_size = KernelPara.fsz;
% filtN = KernelPara.filtN;
% m = filter_size^2 - 1;
% w = repmat(w,[1,filtN]);
% cof_beta = eye(m,m);
% x0 = zeros(length(cof_beta(:)) + 1 + filtN*mfs.NumW, run_stage);
% theta = [10 5 ones(1,run_stage-2)];
% p = [log(1) log(0.1)*ones(1,run_stage-1)];
% 
% for i=1:run_stage
%     x0(:,i) = [cof_beta(:); p(i); w(:)*theta(i)];
% end
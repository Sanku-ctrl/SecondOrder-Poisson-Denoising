clear all;
clc;
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;

global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS DIFF_TERM PT Q_UPPER Q_LOWER
% R = 400;
% path = '/home/staff/cheny/FoETraininigSets/FoETrainingSets256/';
% sigma = 25;
% [G_TRUTH, F_NOISE] = LoadTrainingImages(R, path, sigma);
% INPUT   = F_NOISE;
% DIFF_TERM = zeros(size(INPUT));

% [Q_LOWER, Q_UPPER, G_TRUTH, F_NOISE] = generateDeblockingSamples;
load q10_deblocking_training_samples.mat;
R = 10;
idx = 1:R;
psz = 64;
F_NOISE = noisy(:,idx);
G_TRUTH = clean(:,idx);
Q_UPPER = Q_upper(:,idx);
Q_LOWER = Q_lower(:,idx);
%% pad input images
bsz = 8;
[PT,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;
INPUT   = F_NOISE;
% DIFF_TERM = zeros(size(INPUT));
DIFF_TERM = rand(size(INPUT))*20;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
stage = 1;
[x_init, MFS] = Equal_Initialization(KernelPara, stage);

x0 = x_init(:,1);
n = length(x0);
% x0 = (rand(n,1) - 0.5)*2;
tic;[loss, grad] = loss_with_gradient_unit_filters(x0);toc
tic;[loss_lut, grad_lut] = loss_with_gradient_unit_filters_lut(x0);toc
delta_g = grad_lut - grad;
ratio = delta_g./(abs(grad) + eps);
delta_l = loss_lut - loss;
return;
eps = 1e-6;
check_n = 2089;
grad2 = zeros(check_n,1);
s = 0;
for i = s+1:s+check_n
    xp = x0;
    xp(i) = xp(i) + eps;
    lp = loss_energy_unit_filters(xp);
    
    xm = x0;
    xm(i) = xm(i) - eps;
    lm = loss_energy_unit_filters(xm);
    
    grad2(i) = (lp - lm)/(2*eps);
    fprintf('num: %d, difference = %g\n', i, abs(grad2(i) - grad(i))/(abs(grad(i)) + 1e-9));
end
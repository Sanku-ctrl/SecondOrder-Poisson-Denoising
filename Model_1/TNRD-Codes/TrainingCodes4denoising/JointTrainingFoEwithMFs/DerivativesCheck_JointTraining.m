clear all; close all;
% clc;
addpath('lbfgs');
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;

global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE
load f_new_25;
load g_new;
R = 10;
idx = 1:R;
psz = 64;
F_NOISE = f_noise(:,idx);
G_TRUTH = g_truth(:,idx);
%% pad input images
bsz = 6;
[PT, T,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
STAGE = 5;
[x0, MFS] = Equal_Initialization(KernelPara, STAGE);

[loss_lut, grad_lut] = loss_with_gradient_joint_stages(x0(:));
[loss, grad] = loss_with_gradient_joint_stages_direct(x0(:));

eps = 1e-6;
check_n = 1850;
grad2 = zeros(check_n,1);
s = 576;
for i = s+1:s+check_n
    xp = x0;
    xp(i) = xp(i) + eps;
    lp = loss_joint_stages_direct(xp);
    
    xm = x0;
    xm(i) = xm(i) - eps;
    lm = loss_joint_stages_direct(xm);
    
    grad2(i) = (lp - lm)/(2*eps);
    fprintf('num:%d, grad = %.3f, grad2 = %.3f, difference = %f\n', i, ...
        grad(i), grad2(i), grad2(i) - grad(i));
end



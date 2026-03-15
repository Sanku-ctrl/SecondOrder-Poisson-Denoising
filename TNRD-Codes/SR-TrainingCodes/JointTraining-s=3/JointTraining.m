clear all; close all;
% clc;
addpath('lbfgs');
filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;

global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE SRmtx U0 SCALE
R = 40;
path = '../SR-samples-150-G/';
scale = 3;
[G_TRUTH, U0, F_NOISE] = LoadTrainingImages(R, path, scale);
%% pad input images
psz = 150;
% bsz = filter_size+1;
bsz = 6;
[PT, T, U0, F_NOISE] = PadNoisyPatches(U0, F_NOISE, R, psz, bsz, scale);
PATCH_SIZE = psz + 2*bsz;
SCALE = scale;
% load SRmtx.mat;
load SRmtx-150.mat;
SRmtx = A;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
% training function
fn_foe = @loss_with_gradient_joint_stages;
opts_foe = lbfgs_options('iprint', -1, 'maxits', 250, 'factr', 1e9, ...
    'cb', @test_callback,'m',5,'pgtol', 1e-1);
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
STAGE = 5;
[x0, MFS] = Equal_Initialization(KernelPara, STAGE);

[x,fx,exitflag,userdata] = lbfgs(fn_foe,x0(:),opts_foe);

[len, s] = size(x0);
cof = reshape(x,len,s);
stage = STAGE;
save('JointTraining_91imgs_7x7_stage=5_s=3-G-Bicubic.mat','cof','stage','MFS');


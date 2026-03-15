clear all; close all;
% clc;
addpath('lbfgs');
filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;

global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE Q_UPPER Q_LOWER DCT_MTX
path = '/home/staff/cheny/FoETraininigSets/FoETrainingSets176/';
R = 400;
q = 30;
load flip_mtx.mat;
[Q_LOWER, Q_UPPER, G_TRUTH, F_NOISE] = generateDeblockingSamples(R, path, q, sign_ud, sign_lr, sign_lrud);
load DCT_192.mat;
DCT_MTX{1} = D;
DCT_MTX{2} = D';
%% pad input images
psz = 176;
bsz = 8;
[PT, T, F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
% training function
fn_foe = @loss_with_gradient_joint_stages;
opts_foe = lbfgs_options('iprint', -1, 'maxits', 300, 'factr', 1e9, ...
    'cb', @test_callback,'m',5,'pgtol', 1e-3);
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;

STAGE = 4;
% [x0, MFS] = Equal_Initialization(KernelPara, STAGE);
% load JointTraining_7x7_400_176X176_stage=4.mat;
load JointTraining_7x7_400_176X176_stage=4_Q=20.mat;
x0 = cof;

[x,fx,exitflag,userdata] = lbfgs(fn_foe,x0(:),opts_foe);

[len, s] = size(x0);
cof = reshape(x,len,s);
stage = STAGE;
save('JointTraining_7x7_400_176X176_stage=4_Q=30.mat','cof','stage','MFS');
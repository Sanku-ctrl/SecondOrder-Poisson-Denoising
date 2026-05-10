clear all; close all;
% clc;
addpath('lbfgs');
filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;

global F_NOISE G_TRUTH BASIS PATCH_SIZE KernelPara MFS PT T STAGE
R = 40;
path = '../FoETrainingSets180/';
sigma = 25;
[G_TRUTH, F_NOISE] = LoadTrainingImages(R, path, sigma);
%% pad input images
psz = 180;
bsz = filter_size+1;
[PT, T,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
% training function
fn_foe = @loss_with_gradient_joint_stages;
opts_foe = lbfgs_options('iprint', -1, 'maxits', 300, 'factr', 1e9, ...
    'cb', @test_callback,'m',5,'pgtol', 1e-1);
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
STAGE = 5;
[x0, MFS] = Initialization(KernelPara, STAGE);

[x,fx,exitflag,userdata] = lbfgs(fn_foe,x0(:),opts_foe);

[len, s] = size(x0);
cof = reshape(x,len,s);
stage = STAGE;
save('JointTraining_7x7_400_180x180_stage=5.mat','cof','stage','MFS');

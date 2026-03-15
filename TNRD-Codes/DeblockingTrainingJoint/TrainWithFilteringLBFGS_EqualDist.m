clear all; close all;
% clc;
addpath('lbfgs');
filter_size = 9;
m = filter_size^2 - 1;
filter_num = 40;

global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS DIFF_TERM PT Q_UPPER Q_LOWER DCT_MTX
path = '/home/staff/cheny/FoETraininigSets/FoETrainingSets160/';
R = 300;
q = 10;
load flip_mtx.mat;
[Q_LOWER, Q_UPPER, G_TRUTH, F_NOISE] = generateDeblockingSamples(R, path, q, sign_ud, sign_lr, sign_lrud);
% load q10_deblocking_training_128x128.mat;
load DCT_176.mat;
DCT_MTX{1} = D;
DCT_MTX{2} = D';
%% pad input images
psz = 160;
bsz = 8;
[PT,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;
INPUT   = F_NOISE;
DIFF_TERM = zeros(size(INPUT));

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
% training function
fn_foe = @loss_with_gradient_unit_filters_lut_dct;
opts_foe = lbfgs_options('iprint', -1, 'maxits', 200, 'factr', 1e9, ...
    'cb', @test_callback,'m',5,'pgtol', 1e-3);
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
stage = 6;
[x0, MFS] = Equal_Initialization(KernelPara, stage);

nfoe = length(x0(:,1));
cof = zeros(nfoe,stage);
stage_err = zeros(stage,1);
for s = 1:stage;
    %% step 1: train the FoE model (kernels and weights)
    fprintf('training stage: %d, show initial model\n',s);
%     KernelPara.cof = x0(:,s);
%     show_FOEandMFs(KernelPara, MFS);
    [x,fx,exitflag,userdata] = lbfgs(fn_foe,x0(:,s),opts_foe);
    cof(:,s) = x;
    
    %% do one gradient descent step
    KernelPara.cof = x;
    [loss, deImg] = loss_specific_model(G_TRUTH, INPUT, DIFF_TERM, Q_UPPER, Q_LOWER, KernelPara, MFS, PT, DCT_MTX, 0);
    deImg = ProcessPatches(deImg,R,PATCH_SIZE,bsz);
    DIFF_TERM = deImg - INPUT;
    INPUT = deImg;
    
    stage_err(s) = loss;
    showsamples(6);
    fprintf('stage: %d, training loss = %.3f\n', s, loss);
    save('training_deblocking.mat','cof','stage_err','stage','MFS');
end
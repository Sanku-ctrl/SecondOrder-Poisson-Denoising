clear all; close all;
% clc;
addpath('lbfgs');
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;

global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS T
R = 400;
path = '../FoETrainingSets180/';
sigma = 15;
[G_TRUTH, F_NOISE] = LoadTrainingImages(R, path, sigma);
%% pad input images
psz = 180;
bsz = filter_size+1;
[T,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;
INPUT   = F_NOISE;

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
% training function
fn_foe = @loss_with_gradient_unit_filters_LUT;
opts_foe = lbfgs_options('iprint', -1, 'maxits', 200, 'factr', 1e9, ...
    'cb', @test_callback,'m',5,'pgtol', 1e-3);
%% training algorithm parameters
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
stage = 8;
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
    [loss, deImg] = loss_specific_model(F_NOISE, G_TRUTH, INPUT, KernelPara, MFS, T, 0);
    deImg = ProcessPatches(deImg, R, PATCH_SIZE, bsz);
    INPUT = deImg;
    
    stage_err(s) = loss;
    showsamples(6);
    fprintf('stage: %d, training loss = %.3f\n', s, loss);
    save('training_5x5_400_180_sigma=15.mat','cof','stage_err','stage','MFS');
end
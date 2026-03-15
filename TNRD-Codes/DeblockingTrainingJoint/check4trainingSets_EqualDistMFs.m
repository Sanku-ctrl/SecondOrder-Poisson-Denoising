clear all; close all;
% clc;
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;
global PATCH_SIZE

load q10_deblocking_training_128x128.mat;
load training_deblocking_5x5_128.mat;
R = 400;
idx = 1:R;
F_NOISE = noisy(:,idx);
G_TRUTH = clean(:,idx);
Q_UPPER = Q_upper(:,idx);
Q_LOWER = Q_lower(:,idx);
%% pad input images
psz = 128;
bsz = 8;
[PT,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;
INPUT   = F_NOISE;
DIFF_TERM = zeros(size(INPUT));

BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);

KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% MFs means and precisions
check_stage = 6;
trained_model = save_trained_model(cof, MFS, check_stage, KernelPara);

test_loss = zeros(check_stage,1);
for s = 1:check_stage
    tic;
    [loss, deImg] = deblocking_dataSets(G_TRUTH, INPUT, DIFF_TERM, Q_UPPER, Q_LOWER, trained_model{s}, PT);
    toc
    deImg = ProcessPatches(deImg,R,PATCH_SIZE,bsz);
    DIFF_TERM = deImg - INPUT;
    INPUT = deImg;
    
    test_loss(s) = sum(sum((PT*deImg - G_TRUTH).^2))/R;
end
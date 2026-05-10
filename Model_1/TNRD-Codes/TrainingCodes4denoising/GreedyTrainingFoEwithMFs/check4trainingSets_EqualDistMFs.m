clear all; close all;
% clc;
filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;
global PATCH_SIZE
% load training_process_euqal_dist.mat;
load training_5x5_400_180x180.mat;
% load f_new_25;
% load g_new;
% R = 200;
% idx = 1:R;
% F_NOISE = f_noise(:,idx);
% G_TRUTH = g_truth(:,idx);
R = 400;
path = '/home/staff/cheny/FoETraininigSets/FoETrainingSets180/';
sigma = 25;
[G_TRUTH, F_NOISE] = LoadTrainingImages(R, path, sigma);
%% pad input images
psz = 180;
bsz = 6;
[T,F_NOISE] = PadNoisyPatches(F_NOISE,R,psz,bsz);
PATCH_SIZE = psz + 2*bsz;
INPUT   = F_NOISE;
DIFF_TERM = zeros(size(INPUT));
BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);

KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% MFs means and precisions
trained_model = save_trained_model(cof, MFS, stage, KernelPara);

stage = 8;
test_loss = zeros(stage,1);
for s = 1:stage
    tic;
    [loss, deImg] = CheckRange_specific_model(F_NOISE, G_TRUTH, INPUT, DIFF_TERM, trained_model{s}, T);
    toc
    deImg = ProcessPatches(deImg,R,PATCH_SIZE,bsz);
    DIFF_TERM = deImg - INPUT;
    INPUT = deImg;
    
    test_loss(s) = sum(sum((T*deImg - G_TRUTH).^2))/R;
end
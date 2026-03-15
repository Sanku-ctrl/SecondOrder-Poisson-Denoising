clear all; 
close all;
% clc;
load training_5x5_400_180x180_s=3.mat;

filter_size = 5;
m = filter_size^2 - 1;
filter_num = 24;
BASIS = gen_dct2(filter_size);
BASIS = BASIS(:,2:end);
%% pad and crop operation
up_scale = 3;
bsz = filter_size+1;
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
padLR = @(x) padarray(x,bndry/up_scale,'symmetric','both');
%% MFs means and precisions
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% MFs means and precisions
trained_model = save_trained_model(cof, MFS, stage, KernelPara);
% im = imread('Set14/monarch.bmp');
im = imread('Set5/butterfly_GT.bmp');
%% work on illuminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
im_gnd = modcrop(im, up_scale);
im_gnd = double(im_gnd);
%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');

input = pad(im_b);
noisy = padLR(im_l);

im_gnd = shave(uint8(im_gnd), [up_scale, up_scale]);
im_b = shave(uint8(im_b), [up_scale, up_scale]);
psnr_bic = compute_psnr(im_gnd,im_b);
%% A matrix
data.size_coarse = size(noisy);
data.size_fine = size(input);
param.scale = data.size_fine./data.size_coarse;
A = buildMatrixA(data, param);
%% run denoising, s stages
run_stage = 8;
tic;
for s = 1:run_stage
    deImg = SROneStepGMixMFs(noisy, input, trained_model{s}, A);
    t = crop(deImg);
    deImg = pad(t);
    input = deImg;
    
    %% Evaluation PSNR
    im_h = shave(uint8(t), [up_scale, up_scale]);
    psnr_trd = compute_psnr(im_gnd,im_h);
%     rms = sqrt(mean((t(:) - clean(:)).^2));
    fprintf('stage = %d, BI = %.4f, psnr = %f\n',s, psnr_bic, psnr_trd);
end
toc
%% show images
figure;
% subplot(1,3,1);imshow(im_gnd);title('original image');
% subplot(1,3,2);imshow(im_b);title('BI image');
imshow(im_h);
str = strcat('TRD-SR image,',sprintf('PSNR:%.3f', psnr_trd));
title(str);
drawnow;
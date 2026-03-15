clear all; 
close all;
% clc;
% load JointTraining_91imgs_5x5_stage=5_SR_s=3-G.mat;
% load JointTraining_91imgs_5x5_stage=5_SR_s=3.mat;
% load JointTraining_5x5_stage=8_SR_s=3.mat;
% load JointTraining_91imgs_5x5_stage=5_s=3-G-Bicubic.mat;
load JointTraining_91imgs_7x7_stage=5_s=3-G-Bicubic.mat;

filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;
BASIS = gen_dct2(filter_size);
BASIS = BASIS(:,2:end);
%% pad and crop operation
up_scale = 3;
bsz = 6;
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
I0 = imread('Set14/empress.png');
% I0 = imread('Set5/butterfly_GT.bmp');
%% work on illuminance only
if size(I0,3)>1
    I0 = rgb2ycbcr(I0);
    im = I0(:, :, 1);
end
% im = rgb2gray(im);
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
% A = buildMatrixA(data, param);
%% run denoising, s stages
run_stage = 5;
tic;
for s = 1:run_stage
    deImg = SROneStepGMixMFs(noisy, input, trained_model{s});
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
sr = zeros([size(im_b) 3]);
sr(:,:,1) = im_h;
bi_lu = im_b;
%% color channels 
cb = I0(:, :, 2);
im_gnd = modcrop(cb, up_scale);
im_gnd = single(im_gnd)/255;
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');
im_b = shave(uint8(im_b*255), [up_scale, up_scale]);
sr(:,:,2) = im_b;

cr = I0(:, :, 3);
im_gnd = modcrop(cr, up_scale);
im_gnd = single(im_gnd)/255;
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');
im_b = shave(uint8(im_b*255), [up_scale, up_scale]);
sr(:,:,3) = im_b;

sr = uint8(sr);
bi_sr = sr;
bi_sr(:,:,1) = bi_lu;
rgb = ycbcr2rgb(sr);
%% show images
figure;
% subplot(1,3,1);imshow(im_gnd);title('original image');
% subplot(1,3,2);imshow(im_b);title('BI image');
imshow(rgb);str = strcat('TRD-SR image,',sprintf('PSNR:%.3f', psnr_trd));title(str);drawnow;

rgb = ycbcr2rgb(bi_sr);
figure;imshow(rgb);title('Bicubic Interpolation');

ori = modcrop(ycbcr2rgb(I0), up_scale);
ori = shave(ori, [up_scale, up_scale]);
figure;imshow(ori);title('Original image');
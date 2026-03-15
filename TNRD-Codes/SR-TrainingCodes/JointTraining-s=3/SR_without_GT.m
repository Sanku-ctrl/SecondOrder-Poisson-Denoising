clear all; 
close all;
% clc;
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
I0 = imread('Set14/Picture2.png');
%% work on illuminance only
if size(I0,3)>1
    I0 = rgb2ycbcr(I0);
    im = I0(:, :, 1);
end
%% bicubic interpolation
im_l = double(im);
im_b = imresize(im_l, up_scale, 'bicubic');
im_n = imresize(im_l, up_scale, 'nearest');

input = pad(im_b);
noisy = padLR(im_l);
im_b = shave(uint8(im_b), [up_scale, up_scale]);
%% A matrix
data.size_coarse = size(noisy);
data.size_fine = size(input);
param.scale = data.size_fine./data.size_coarse;
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
    fprintf('stage = %d\n',s);
end
toc
sr = zeros([size(im_b) 3]);
sr_n = zeros([size(im_b) 3]);
sr(:,:,1) = im_h;
sr_n(:,:,1) = shave(uint8(im_n), [up_scale, up_scale]);
bi_lu = im_b;
%% color channels 
cb = I0(:, :, 2);
im_l = double(cb);
im_b = imresize(im_l, up_scale, 'bicubic');
im_b = shave(uint8(im_b), [up_scale, up_scale]);
sr(:,:,2) = im_b;
sr_n(:,:,2) = shave(uint8(imresize(im_l, up_scale, 'nearest')), [up_scale, up_scale]);

cr = I0(:, :, 3);
im_l = double(cr);
im_b = imresize(im_l, up_scale, 'bicubic');
im_b = shave(uint8(im_b), [up_scale, up_scale]);
sr(:,:,3) = im_b;
sr_n(:,:,3) = shave(uint8(imresize(im_l, up_scale, 'nearest')), [up_scale, up_scale]);

sr = uint8(sr);
bi_sr = sr;
bi_sr(:,:,1) = bi_lu;
rgb = ycbcr2rgb(sr);
%% show images
figure;
imshow(rgb);str = strcat('TRD-SR image');title(str);drawnow;

rgb = ycbcr2rgb(bi_sr);
figure;imshow(rgb);title('Bicubic Interpolation');

rgb = ycbcr2rgb(uint8(sr_n));
figure;imshow(rgb);title('Nearest Interpolation');
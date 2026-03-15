clear all; 
% close all;
% clc;
% load training_5x5_400_180x180.mat;
% load Training_5x5_400_256x256.mat;
% load training_5x5_400_180x180_truncation.mat;
% load training_process_7x7_180.mat;
load training_7x7_180_newRBFs_quantized_noisy.mat;
filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;
BASIS = gen_dct2(filter_size);
BASIS = BASIS(:,2:end);
%% pad and crop operation
bsz = 8;
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
%% MFs means and precisions
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% MFs means and precisions
trained_model = save_trained_model(cof, MFS, stage, KernelPara);
%% test 68 images
img_num = 68;
sigma = 25;
psnr = zeros(img_num,1);
time = zeros(img_num,1);
reset(RandStream.getDefaultStream);
for img_idx = 1:img_num
path = './68imgs/';
file = strcat(path,sprintf('test%03d.png', img_idx));
I0 = double(imread(file));
% Im = I0 + sigma*randn(size(I0));
path2 = '/home/staff/cheny/BSDS300/images/68imgs/';
file = strcat(path2,sprintf('test%03d_025.png', img_idx));
Im = double(imread(file));

[R,C] = size(I0);
rms1 = sqrt(mean((Im(:) - I0(:)).^2)) /sigma * 25
%% run denoising, 15 stages
input = pad(Im);
noisy = pad(Im);
clean = I0;
diffterm = zeros(size(input));
run_stage = 6;
tic;
for s = 1:run_stage
    deImg = denoisingOneStepGMixMFs(noisy, input, diffterm, trained_model{s});
    t = crop(deImg);
    deImg = pad(t);
    diffterm = deImg - input;
    input = deImg;
    
    rms = sqrt(mean((t(:) - clean(:)).^2));
    fprintf('stage = %d, psnr = %f\n',s, 20*log10(255/rms));
end
time(img_idx) = toc;
x_star = t(:);
%% recover image
rms2 = sqrt(mean((x_star - I0(:)).^2));
PSNR = 20*log10(255/rms2);
recover = reshape(x_star,R,C);
fprintf('Denoising image %3d\tPSNR: %.3f\n',img_idx,PSNR);
%% show images
% figure(img_idx);
% subplot(1,3,1);imshow(I0,[0 255]);title('original image');
% subplot(1,3,2);imshow(Im,[0 255]);title('noisy image');
% subplot(1,3,3);imshow(recover,[0 255]);
% str = strcat('denoised image,',sprintf('PSNR:%.3f', PSNR));
% title(str);
% drawnow;
psnr(img_idx) = PSNR;
end
mean(psnr)